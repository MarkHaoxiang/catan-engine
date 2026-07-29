"""The planner: a stateful decision-tree agent over plain Python logic.

One :class:`~settlrl_agents.planner.tree.Selector` of leaf handlers covers the
forced phases (setup, discard, robber, trade response, roll); the MAIN-phase
pair carries the strategy: ``ExecutePlan`` keeps a persistent build plan
(chosen by scoring city upgrades, road-path-plus-settlement expansions, dev
buys, and a road filler) and plays its next step the moment it is legal;
``Acquire`` spends the turn closing the plan's resource gap (dev-card plays,
maritime trades at our best port ratio, capped domestic proposals with a
rejected-offer memory). Everything reads legality off the mask, so the tree
never re-implements a rule; an unexpected state just falls through to a
fixed-priority fallback.
"""

from __future__ import annotations

import random

import numpy as np
from settlrl_engine.board.resources import N_RESOURCES
from settlrl_engine.board.state import GamePhase
from settlrl_engine.env import ActionType
from settlrl_engine.mechanics.trade import pack_trade_single
from settlrl_search.policy import (
    GameAgent,
    HostFlatMask,
    HostObservation,
    StatefulPolicy,
)

from settlrl_agents.planner.goals import RES_WEIGHT, choose_plan, plan_candidates, wprod
from settlrl_agents.planner.pov import (
    COST_CITY,
    COST_DEV,
    COST_ROAD,
    COST_SETTLEMENT,
    EDGE_ENDPOINTS,
    ROW_IDX,
    ROW_TARGET,
    ROWS_OF_TYPE,
    VERTEX_NEIGHBORS,
    Pov,
    flat_row,
)
from settlrl_agents.planner.tactic import Tactic
from settlrl_agents.planner.tree import Blackboard, Node, Selector

# Own turns a plan may sit with no step realized before it is abandoned
# (covers goals starved by piece limits or an empty dev deck, which the
# observation cannot rule out up front; routine adaptation is the rival-goal
# switch in ExecutePlan, so this is only the starvation backstop).
_PLAN_PATIENCE = 8


class SetupSettlement(Node):
    """Best legal starting spot: production value plus a tactical tie-break.

    The successor value sees pips, ports and the two-sided position; the
    scripted terms add what one ply cannot — new-resource coverage and the
    surrounding expansion room."""

    def __init__(self, tactic: Tactic) -> None:
        self.tactic = tactic

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.SETUP_SETTLEMENT)
        if rows.size == 0:
            return None
        vals = self.tactic.values(pov)
        best, best_score = int(rows[0]), -np.inf
        for row in rows:
            v = int(ROW_IDX[row])
            prod = pov.vertex_production(v)
            new_types = float(RES_WEIGHT[(prod > 0) & (pov.my_production == 0)].sum())
            around = max(
                (
                    wprod(pov.vertex_production(n))
                    for n in VERTEX_NEIGHBORS[v]
                    if int(pov.vertex_owner[n]) == 0
                ),
                default=0.0,
            )
            score = float(vals[row]) + 2.0 * new_types + 0.2 * around + bb.noise()
            if score > best_score:
                best, best_score = int(row), score
        bb.last_setup_vertex = int(ROW_IDX[best])
        return best


class SetupRoad(Node):
    """Point the free road toward the most promising next settlement spot."""

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.SETUP_ROAD)
        if rows.size == 0:
            return None
        best, best_score = int(rows[0]), -np.inf
        for row in rows:
            e = int(ROW_IDX[row])
            a, b = int(EDGE_ENDPOINTS[e, 0]), int(EDGE_ENDPOINTS[e, 1])
            far = b if a == bb.last_setup_vertex else a
            # Aim at the best spot within two further roads of the far end.
            ring = {far} | set(VERTEX_NEIGHBORS[far])
            ring |= {m for n in ring.copy() for m in VERTEX_NEIGHBORS[n]}
            potential = max(
                (
                    wprod(pov.vertex_production(n))
                    * (1.0 if n in VERTEX_NEIGHBORS[far] else 0.6)
                    for n in ring
                    if pov.settleable(n)
                ),
                default=0.0,
            )
            score = potential + 0.3 * wprod(pov.vertex_production(far))
            score += bb.noise()
            if score > best_score:
                best, best_score = int(row), score
        return best


class DiscardSurplus(Node):
    """Give up the card whose loss the successor value minds least."""

    def __init__(self, tactic: Tactic) -> None:
        self.tactic = tactic

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.DISCARD)
        if rows.size == 0:
            return None
        return self.tactic.best(pov, rows)


class MoveRobber(Node):
    def __init__(self, tactic: Tactic) -> None:
        self.tactic = tactic

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.MOVE_ROBBER)
        return self.tactic.best_paranoid(pov, rows) if rows.size else None


class PlayKnight(Node):
    """The pre-roll knight (legal before rolling, rulebook p.7), decided by
    expectimax: play it exactly when the best relocation raises the exact
    11-outcome expectation of our own pending roll — which prices unblocking,
    denial, the steal, and the army race in one comparison. Post-roll, only
    an immediate Largest Army grab fires here (DenialKnight owns the rest)."""

    def __init__(self, tactic: Tactic) -> None:
        self.tactic = tactic

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        rows = pov.legal_rows(ActionType.PLAY_KNIGHT)
        if rows.size == 0:
            return None
        if pov.phase == GamePhase.ROLL and pov.current_player == pov.me:
            best = self.tactic.best_paranoid(pov, rows)
            gain = self.tactic.roll_expectation(
                pov, best
            ) - self.tactic.roll_expectation(pov)
            return best if gain > 0.3 else None
        after = int(pov.knights_played[pov.me]) + 1
        others = np.delete(pov.knights_played, pov.me)
        takes_army = (
            pov.largest_army_owner != pov.me
            and after >= 3
            and after > int(others.max())
        )
        return self.tactic.best_paranoid(pov, rows) if takes_army else None


class RespondToTrade(Node):
    """Accept exactly the offers whose successor values better than refusing."""

    def __init__(self, tactic: Tactic) -> None:
        self.tactic = tactic

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        accept = pov.legal(ActionType.ACCEPT_TRADE)
        reject = pov.legal(ActionType.REJECT_TRADE)
        if accept is None and reject is None:
            return None
        if accept is None or reject is None:
            return accept if reject is None else reject
        vals = self.tactic.values(pov)
        return accept if vals[accept] > vals[reject] else reject


class RollDice(Node):
    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        return pov.legal(ActionType.ROLL_DICE)


class PlayForced(Node):
    """The committed second half of a combo, played while it is still legal."""

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        row, bb.forced_row = bb.forced_row, None
        if row is not None and pov.mask[row]:
            return row
        return None


class ExecutePlan(Node):
    """Keep a live plan and play its next step the moment it is legal."""

    def __init__(self, depth: int) -> None:
        self.depth = depth

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        if not pov.my_turn_main:
            return None
        stale = bb.plan is not None and bb.plan_age > _PLAN_PATIENCE
        if (
            bb.plan is None
            or stale
            or bb.plan.invalid(pov)
            or bb.plan.next_step(pov) is None
        ):
            exclude = bb.plan.name if stale and bb.plan is not None else None
            bb.set_plan(choose_plan(pov, bb, self.depth, exclude))
        elif bb.plan_age >= 1:
            # Stay adaptive without thrashing: a freshly-scored rival goal
            # replaces the incumbent only on a clear margin (>> the noise).
            cands = plan_candidates(pov, bb, self.depth)
            current = next((s for s, p in cands if p.name == bb.plan.name), None)
            best_score, best_plan = max(cands, key=lambda c: c[0])
            if current is not None and best_score > current + 2.5:
                bb.set_plan(best_plan)
        if bb.plan is None:
            return None
        step = bb.plan.next_step(pov)
        if step is None:
            return None
        if step.kind == ActionType.BUY_DEVELOPMENT_CARD:
            return pov.legal(ActionType.BUY_DEVELOPMENT_CARD)
        return pov.legal(step.kind, step.idx)


class OpportunisticBuild(Node):
    """Spend pure surplus on anything the plan is not waiting for — an extra
    city, settlement, road, or dev buy — when its successor value clears the
    do-nothing baseline (END_TURN's successor). Banked progress beats robber
    bait, and the value sees what a static rank cannot (breaking an
    opponent's road, opening a spot, the award races)."""

    _COSTS = (
        (COST_CITY, ActionType.BUILD_CITY),
        (COST_SETTLEMENT, ActionType.BUILD_SETTLEMENT),
        (COST_ROAD, ActionType.BUILD_ROAD),
        (COST_DEV, ActionType.BUY_DEVELOPMENT_CARD),
    )

    def __init__(self, tactic: Tactic) -> None:
        self.tactic = tactic

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        if not pov.my_turn_main or bb.plan is None:
            return None
        end_turn = pov.legal(ActionType.END_TURN)
        if end_turn is None:
            return None
        surplus = pov.hand - bb.plan.reserved(pov)
        rows: list[int] = []
        for cost, atype in self._COSTS:
            if bool(np.all(surplus >= cost)):
                rows.extend(int(r) for r in pov.legal_rows(atype))
        if not rows:
            return None
        vals = self.tactic.values(pov)
        above = [r for r in rows if vals[r] > vals[end_turn]]
        if not above:
            return None
        return self.tactic.best_paranoid(pov, np.asarray(above))


_ROW_IS_BUILDISH = np.zeros(int(ROW_IDX.shape[0]), dtype=bool)
for _t in (
    ActionType.BUILD_CITY,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUY_DEVELOPMENT_CARD,
):
    _ROW_IS_BUILDISH[ROWS_OF_TYPE[int(_t)]] = True


class EnablerCombo(Node):
    """Own-turn two-ply tactic lookahead alone cannot see: a maritime trade
    or Year of Plenty that *enables* a city / settlement / dev buy this same
    turn, judged by the pair's final value against just ending the turn. The
    follow-up is committed on the blackboard and played next tick."""

    def __init__(self, tactic: Tactic) -> None:
        self.tactic = tactic

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        if not pov.my_turn_main:
            return None
        end_turn = pov.legal(ActionType.END_TURN)
        enablers = [int(r) for r in pov.legal_rows(ActionType.MARITIME_TRADE)]
        enablers += [int(r) for r in pov.legal_rows(ActionType.PLAY_YEAR_OF_PLENTY)]
        if end_turn is None or not enablers:
            return None
        pair = self.tactic.combo_best(pov, enablers, _ROW_IS_BUILDISH)
        if pair is None:
            return None
        enabler, follow, value = pair
        if value <= self.tactic.values(pov)[end_turn] + 0.5:
            return None
        bb.forced_row = follow
        return enabler


class Acquire(Node):
    """Close the plan's resource gap: dev-card plays, then the bank, then a
    domestic proposal (capped per turn, never re-offering a rejected one)."""

    def __init__(self, max_proposals: int) -> None:
        self.max_proposals = max_proposals

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        if not pov.my_turn_main or bb.plan is None:
            return None
        surplus = pov.hand - bb.plan.reserved(pov)
        # Pure surplus covering a whole dev card buys one on the spot: it
        # never starves the plan, and idle cards are robber bait.
        if bool(np.all(surplus >= COST_DEV)):
            row = pov.legal(ActionType.BUY_DEVELOPMENT_CARD)
            if row is not None:
                return row
        need = bb.plan.need(pov)
        if need.sum() == 0:
            return None
        by_need = [int(r) for r in np.argsort(-need) if need[r] > 0]

        rows = pov.legal_rows(ActionType.PLAY_YEAR_OF_PLENTY)
        if rows.size:
            best = max(rows, key=lambda r: int(need[ROW_IDX[r]] + need[ROW_TARGET[r]]))
            if need[ROW_IDX[best]] + need[ROW_TARGET[best]] >= 1:
                return int(best)

        rows = pov.legal_rows(ActionType.PLAY_MONOPOLY)
        if rows.size:
            opp = (19 - pov.bank) - pov.hand  # cards of each type in other hands
            best = max(
                rows,
                key=lambda r: int(opp[ROW_IDX[r]]) + 2 * (need[ROW_IDX[r]] > 0),
            )
            r = int(ROW_IDX[best])
            # Worth the dev-play slot when it feeds the plan or is a mass
            # grab (excess converts at the bank).
            if (need[r] > 0 and opp[r] >= 2) or opp[r] >= 4:
                return int(best)

        road_steps = sum(
            1
            for s in bb.plan.steps
            if s.kind == ActionType.BUILD_ROAD and not s.realized(pov)
        )
        if road_steps >= 1:
            row = pov.legal(ActionType.PLAY_ROAD_BUILDING)
            if row is not None:
                return row

        best_row, best_score = None, -np.inf
        for g in range(N_RESOURCES):
            if need[g] > 0 or surplus[g] < pov.port_ratio[g]:
                continue
            for r in by_need:
                row = pov.legal(ActionType.MARITIME_TRADE, g, r)
                if row is None:
                    continue
                score = 10.0 * float(need[r]) + float(surplus[g])
                if score > best_score:
                    best_row, best_score = row, score
        if best_row is not None:
            return best_row

        if (
            pov.n_players > 2
            and bb.proposals_this_turn < self.max_proposals
            and bb.pending_proposal is None
        ):
            partners = sorted(
                (p for p in range(pov.n_players) if p != pov.me),
                key=lambda p: -int(pov.hand_size[p]),
            )
            for r in by_need:
                for g in range(N_RESOURCES):
                    if need[g] > 0 or surplus[g] < 1:
                        continue
                    for partner in partners:
                        if (g, r, partner) in bb.rejected:
                            continue
                        idx, target = pack_trade_single(g, r, partner)
                        row = flat_row(ActionType.PROPOSE_TRADE, int(idx), int(target))
                        if not pov.mask[row]:
                            continue
                        bb.pending_proposal = (g, r, partner, pov.hand.copy())
                        bb.proposals_this_turn += 1
                        return row
        return None


class SpendDown(Node):
    """Ending the turn with more than seven cards is half-a-hand of robber
    exposure: bank the excess in a dev card first, plan reservations or not."""

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        if not pov.my_turn_main or int(pov.hand.sum()) <= 7:
            return None
        return pov.legal(ActionType.BUY_DEVELOPMENT_CARD)


class DenialKnight(Node):
    """End-of-turn knight, value-timed: play it when the best relocation's
    successor (denial, the steal, the Largest Army resolve — all inside one
    apply) clearly beats just ending the turn."""

    def __init__(self, tactic: Tactic) -> None:
        self.tactic = tactic

    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        if not pov.my_turn_main:
            return None
        rows = pov.legal_rows(ActionType.PLAY_KNIGHT)
        end_turn = pov.legal(ActionType.END_TURN)
        if rows.size == 0 or end_turn is None:
            return None
        vals = self.tactic.values(pov)
        best = int(rows[int(np.argmax(vals[rows]))])
        return best if vals[best] > vals[end_turn] + 0.3 else None


class EndTurn(Node):
    def tick(self, pov: Pov, bb: Blackboard) -> int | None:
        return pov.legal(ActionType.END_TURN)


# Fixed action-type priority for states the tree declines on; PROPOSE_TRADE
# is deliberately absent (an unmanaged proposal could re-offer forever).
_FALLBACK_ORDER = (
    ActionType.ROLL_DICE,
    ActionType.SETUP_SETTLEMENT,
    ActionType.SETUP_ROAD,
    ActionType.DISCARD,
    ActionType.MOVE_ROBBER,
    ActionType.REJECT_TRADE,
    ActionType.ACCEPT_TRADE,
    ActionType.BUILD_CITY,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_ROAD,
    ActionType.BUY_DEVELOPMENT_CARD,
    ActionType.END_TURN,
    ActionType.MARITIME_TRADE,
    ActionType.PLAY_KNIGHT,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_MONOPOLY,
)


def _fallback(pov: Pov) -> int:
    for atype in _FALLBACK_ORDER:
        rows = pov.legal_rows(atype)
        if rows.size:
            return int(rows[0])
    return 0


class PlannerAgent:
    """One seat's stateful decision tree (a :class:`GameAgent`)."""

    def __init__(
        self,
        seed: int,
        *,
        expansion_depth: int = 3,
        max_proposals_per_turn: int = 2,
    ) -> None:
        self._bb = Blackboard(rng=random.Random(seed))
        self._was_my_roll = False
        tactic = Tactic(seed)
        self._root = Selector(
            SetupSettlement(tactic),
            SetupRoad(),
            DiscardSurplus(tactic),
            MoveRobber(tactic),
            RespondToTrade(tactic),
            PlayKnight(tactic),
            RollDice(),
            PlayForced(),
            ExecutePlan(expansion_depth),
            OpportunisticBuild(tactic),
            EnablerCombo(tactic),
            Acquire(max_proposals_per_turn),
            SpendDown(),
            DenialKnight(tactic),
            EndTurn(),
        )

    def _sync(self, pov: Pov) -> None:
        """Per-turn memory upkeep: reset on our new turn, and mark the last
        proposal rejected if our hand shows no trace of it."""
        my_roll = pov.phase == GamePhase.ROLL and pov.current_player == pov.me
        if my_roll and not self._was_my_roll:
            self._bb.begin_turn()
        self._was_my_roll = my_roll
        if self._bb.pending_proposal is not None and pov.my_turn_main:
            g, r, partner, before = self._bb.pending_proposal
            if pov.hand[r] <= before[r]:
                self._bb.rejected.add((g, r, partner))
            self._bb.pending_proposal = None

    def act(self, obs: HostObservation, mask: HostFlatMask) -> int:
        pov = Pov(obs, mask)
        if not pov.mask.any():
            return 0
        self._sync(pov)
        row = self._root.tick(pov, self._bb)
        if row is None or not pov.mask[row]:
            row = _fallback(pov)
        return int(row)


def make_planner(
    *, expansion_depth: int = 3, max_proposals_per_turn: int = 2
) -> StatefulPolicy:
    """The planner family: ``expansion_depth`` bounds how many new roads a
    settlement plan may path through; ``max_proposals_per_turn`` caps domestic
    offers. Returns the per-game agent factory."""

    def build(seed: int) -> GameAgent:
        return PlannerAgent(
            seed,
            expansion_depth=expansion_depth,
            max_proposals_per_turn=max_proposals_per_turn,
        )

    return build
