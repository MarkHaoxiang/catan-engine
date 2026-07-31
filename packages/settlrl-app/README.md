# settlrl-app

The Settlrl game server. FastAPI serves the game over a JSON + SSE API, runs the
async game runtime, persists games and accounts, and serves the SPA; a Vite +
React + TypeScript frontend renders the board as SVG. The game model itself is
`settlrl-game`; bot moves are delegated to a remote bot service
(`settlrl-agents[service]`). The app depends on neither the engine nor the
agents directly.

A menu lets you choose between modes, each at its own URL. **Play** opens the
[lobby](#lobby) — the hub to host a game, join an open one, or Quick Match.

A game (`/play/{id}`) is a live, playable game. Each seat is configured per game as a human sharing the screen (hotseat) or a bot (a `settlrl-agents` policy); with no human seats the game plays itself as a spectated bot match. Search bots expose their build parameters, like `mcts`'s simulation budget, behind the seat's gear button. The interface is the table itself:

- **Board** — every placement you can currently make is marked in your colour (hover to preview the piece); click it and confirm in a popup that shows the build cost or who to rob.
- **Hand** — click a glowing development card to play it (resource picks come from a popover), or resource cards to discard after a 7; a knight can also be played straight off the robber pawn, which lights up to take the click.
- **Trading** — click a bank resource pile to maritime-trade at your best port rate, or an opponent's hand pile to compose a 1:1 offer they answer on their turn; buy a development card from the bank's deck.
- **Table scene** — top-down, with zoom, pan, and spin (mouse, touch, or keyboard: arrows pan, `+`/`−` zoom, `[` `]` spin, `0` re-fits). When you hold a single seat the table opens rotated to face it (your play area at the bottom); spectating or a shared-screen hotseat keeps the canonical bottom-facing view. The bank's card piles sit left of the board; each seat's play area lines its table edge with its face-down hand and dev piles and unbuilt roads, settlements, and cities. The dice rest in the corner beside the acting seat: they glow gold to roll, then glow red as the end-turn control once you've rolled.
- **Trade answers** — a trade offered to you shows as a card over the board with Accept / Reject.
- **Bars and panels** — the bottom panel shows the acting human's hand and a status line; the top bar holds New game and a light/dark theme toggle. The right column lists the seats in playing order — each with its holder (a signed-in player's account name, "Guest" for an anonymous human, or the bot kind), points, cards, devs, and 🔍 to inspect an opponent's proven hand bounds — above a chat panel that doubles as the game log.
- **New game** — the top bar's New game button (and the end-of-game screen) returns to the [lobby](#lobby), where hosting opens a [lobby room](#lobby) to set up and start a fresh game. The win target defaults to the player count — **15 victory points for a 2-player duel, 10 otherwise** (shown by the board when it isn't the usual 10) — but the host can set any target; it's recorded with the game so a replay ends where the original did.

A help page (`/help`, the **?** button) documents the controls and icons.

**Replay** (`/replay`) — step through a recorded game. Load a saved game-record file (the JSON from `GET /api/games/{id}/record`) or the live game so far, then scrub with the slider, step move by move, or press play; the log fills in as the game advances, and the record can be saved back to a file.

Each game is a plain-Python `settlrl-game` game; the server holds many live
games at once, addressed by id.
Claiming a human seat (hosting or joining) issues a bearer token, and every request
proves its seats via the `X-Seat-Tokens` header: snapshots are per-seat views — your own
hand arrives in full, everyone else's only as public counts, and the legal-move list only
ships to the seat whose turn it is. You keep your seats through your signed-in account or
this browser's storage.
Pre-game staging is a **[lobby](#lobby)** (`/lobby/{id}`), not a game: a lobby holds
configuration and membership only, the room renders a board *preview* from the seed, and
the engine is built exactly once, when the host starts it — there is no game id until then.
An **online** lobby's human seats start open for others to join (the 🔗 button copies the
invite link) and it can be **listed** in the public lobby (below) for anyone to join; a
**hotseat** lobby claims every human seat for the host's screen, so it can start at once.
Start is host-only and refuses while any human seat is still open, so a half-empty table
can never begin; on start the claimed seats — tokens and all — and the chat carry into the
fresh game and everyone is sent to `/play/{id}`.
The server
pushes state: each client holds an event stream (`GET /api/games/{id}/events`, SSE) and
receives its per-seat snapshot on every change, and bot seats are played by a server-side
driver pacing one move at a time so each lands as its own pushed snapshot — games advance
with no tab open, and every move animates.

## Requirements

- Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18

## Development

The quickest start is the dev launcher, which boots the API, a couple of one-bot
services, and the frontend together, pre-registers an admin account
(`dev@example.com` / `devpassword`), and registers the bot services so bots are
seatable on first load. Ctrl-C stops everything.

```bash
./packages/settlrl-app/dev.sh
```

To run the pieces by hand instead, start the API server and the frontend dev
server in separate terminals from the repo root.

**Terminal 1 — API (port 8000)**
```bash
uv run settlrl-app
```

The server runs JAX on CPU by default (one live game doesn't need a GPU, and JAX would
otherwise preallocate most of its memory). Set `JAX_PLATFORMS=cuda` to override.

**Terminal 2 — frontend (port 5173)**
```bash
cd packages/settlrl-app/frontend
npm install   # first time only
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to the FastAPI server, so hot-reload and the API work together out of the box.

## Production build

Build the frontend into `frontend/dist/`, then start the API server — it detects the built assets and serves them automatically.

```bash
cd packages/settlrl-app/frontend && npm run build
uv run settlrl-app
```

Open http://localhost:8000.

## Hosting

The server is configured by environment variables: `HOST` (default `0.0.0.0`),
`PORT` (default `8000`), `RELOAD` (default `1`; set `0` in production — the
reloader is a dev file-watcher),
`SETTLRL_APP_STATE_DIR` (a directory to persist games in — see below),
`SETTLRL_APP_TURN_TIMEOUT_S` (default `0` = off; after this many seconds of an
idle human turn the server auto-plays a move, so an abandoned game still
finishes instead of stalling), `SETTLRL_APP_MAX_ACTIVE` (default `16`; games
running at once before new creators are queued — keep it below the registry
cap), `SETTLRL_APP_BOT_PROVIDERS` (comma-separated bot-service base URLs
registered on startup — retried until each is reachable — so their bots are
seatable without a manual admin call), and `ROOT_PATH` (the proxy prefix when
served under a path). Run **one process, one worker**: live games are held in
memory, so extra workers would split them.
The registry holds up to 32 games, evicting finished games, hour-idle ones, or
unstarted ones idle past a few minutes (so a burst of empty games can't pin
every slot) to make room.

**Persistence.** Without `SETTLRL_APP_STATE_DIR`, games live only in memory and
a restart loses them. Point it at a (mounted) directory and each game is
journalled — its setup plus every move, seat claim, and chat line — into the same
SQLite database as accounts (`settlrl.db` there) and replayed back into the
registry on the next startup, so a deploy or crash resumes games in progress,
seat tokens and all. Bot pacing restarts for resumed games. Evicted games are
dropped from the database.

Anyone can create games; the concurrency cap queues them past
`SETTLRL_APP_MAX_ACTIVE`. For a public deployment, front the server with a
proxy that rate-limits — the built-in caps (a 2 MB request-body limit, a replay
move-count cap, and high-entropy game ids) bound resource use but are not a
substitute for one.

The repo-root `Dockerfile` builds a self-contained image (frontend compiled
in, CPU JAX):

```bash
docker build -t settlrl-app .
docker run -p 8000:8000 settlrl-app
```

To serve under a path instead of a (sub)domain — e.g. `markhaoxiang.com/settlrl`
behind a proxy that strips the prefix (Caddy `handle_path /settlrl/*`) — bake
the prefix into the frontend and tell FastAPI about it:

```bash
docker build -t settlrl-app --build-arg BASE_PATH=/settlrl/ .
docker run -p 8000:8000 -e ROOT_PATH=/settlrl settlrl-app
```

The mark-haoxiang repo's `infra/` wires this up as the `settlrl` compose
service behind its Caddy.

Seat tokens are bearer secrets, so put TLS in front for anything beyond a
LAN — e.g. Caddy, which manages certificates itself:

```
games.example.com {
    reverse_proxy localhost:8000
}
```

## Accounts

Accounts are optional: anonymous play — claim a seat, get a per-seat token —
works without one. Registering gives a player a persistent identity and lets an
operator mark some users as **admins**, who manage the bot services below. In
the UI the menu's **Sign in** link opens the account page (`/login`,
`/register`).

Accounts are handled by [fastapi-users](https://fastapi-users.github.io/fastapi-users/).
Login uses the OAuth2 password flow (`POST /api/auth/login` returns a bearer
token presented as `Authorization: Bearer …`); tokens are stored server-side, so
`POST /api/auth/logout` truly revokes one. Accounts, tokens, and games all share
the one SQLite database (`settlrl.db` under the state dir, or
`SETTLRL_APP_USER_DB`). Emails listed in `SETTLRL_APP_ADMIN_EMAILS`
(comma-separated) are granted admin on register and login. Endpoints:
`POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`,
`GET /api/users/me`.

**Seats follow the account.** A seat claimed while signed in is tied to your
user id, not just the per-device seat token, so you are recognised — and can
resume your games (`GET /api/me/games`) — on any device without carrying the
token. Send the bearer token alongside (or instead of) `X-Seat-Tokens`; each
snapshot's `your_seats` lists the seats the requester owns either way.

**Profile and history.** The account link opens the **profile** page
(`/profile`): your in-progress games plus a history of finished ones
(`GET /api/me/history`), each replayable or downloadable by id. Finished games
are kept in the store (a capped, replayable archive) rather than discarded, so
their record is served even after the live game is evicted — the same record the
end-of-game screen's **Download replay** button saves.

## Lobby

The menu's **Play** card opens the **Lobby** page (`/lobby`, public), the entry
point for online play: **Host a game**, **Quick Match**, or join an open lobby.
The open list shows the lobbies created with **List in lobby** that still have
an open human seat, newest first (`GET /api/lobbies`); each row shows its player
count, open seats, and map, and **Join** claims a free seat and opens the room.

**Host a game** opens a **lobby** (`POST /api/lobbies`) and drops you into its
**lobby room** (`/lobby/{id}`) — a three-column page (players, the map and
settings, and chat) where the table is staged. A lobby holds configuration and
membership, never a game: the room renders a board preview from the seed, and
the host's edits (`POST /api/lobbies/{id}/configure`, host-only) change the map
(seed / number placement), player count, win target, and per-seat human-or-bot,
and toggle **List in lobby** / **Open to Quick Match** — all streamed to every
participant, with the seat claims and the chat preserved across each change. A
lobby is either **online** (human seats start open and others join them) or
**hotseat** (the host's browser drives every human seat, so it can start at
once). Listing a lobby publicly requires a signed-in account (`listed` returns
`401` otherwise); anonymous play stays invite-link only. **Start game**
(`POST /api/lobbies/{id}/start`, host-only) refuses while any human seat is
still open (`409`) — seat a bot there or wait for a joiner — then builds the
engine, carries the claimed seats (tokens and all) and the chat into the fresh
game, and everyone is sent into `/play/{id}`. A listed lobby the host also
marks `searchable` shows a **Quick Match** tag in its row (a visibility flag;
the matchmaker still forms its own games).

**Quick Match** pairs you into a game with one click (`POST /api/matchmake`,
2- or 4-player). Waiters are pooled per player count and matched on their
[leaderboard](#leaderboard) rating — signed-in players by their real Elo,
anonymous players at a fresh rating — and the rest of the table is filled with
the registered bots whose rating sits closest to the group's. The Elo window
widens the longer you wait, and once the oldest waiter has waited long enough a
game forms with bots so no one is ever stuck. Polling re-presents a ticket until
a seat comes back (like the create-queue) — no new push surface.

## Leaderboard

The menu's **Leaderboard** page (`/leaderboard`, public) ranks players by skill,
**split by player count** — a 2-, 3-, and 4-player ladder are kept separately.
Both registered accounts (shown by their handle) and bots (shown by name) are
rated on the same ladders, so games against bots count. Rating uses
[openskill](https://github.com/vivekjoshy/openskill.py) (the patent-free
Weng-Lin / Plackett-Luce model): when a game ends its final standings are scored
winner-takes-all (the winner ranks first, the rest tie behind) and the whole
table is rated in one step. The displayed number is the conservative ordinal
scaled to an Elo-like range (a fresh player shows ~1000). A seat that is neither
an account nor a bot (an anonymous, token-only human) is not rated, and a bot
occupying more than one seat in a game is skipped for that game. Endpoint:
`GET /api/leaderboard`.

## Admin

A superuser-only status page (`/admin`, reached from the **⚙ Admin** link pinned
bottom-right for admins) shows the running server at a glance: uptime, the live
game count against the registry cap, the registered bot services and seatable
kinds, and a table of the current games (id, players, phase, moves, open seats,
listed / Quick-Match flags, age). It polls `GET /api/admin/status` (superuser
only — `401`/`403` otherwise); the bottom-right link and the page itself are
gated on the account's superuser flag. The **bot services** are also *managed*
here — register one by base URL or remove one (the [Bot services](#bot-services)
controls live on this page, not the menu).

## Bot services

The game server runs **no** agent code in-process. A seat's bot moves are
computed by a separate **bot service** — each service hosts **one bot**
(`settlrl-agents[service]`), deployed and scaled apart from the game server. It
speaks a standardized, structured two-call API:

- `GET /info` — the bot's identity (`{ name, title, description, counts }`).
- `POST /act` — given a game's setup and the moves the service has not seen yet
  (structured `MoveModel`s in board coordinates, the tail after a `base` cursor),
  it advances the game it is tracking and returns the acting seat's chosen move.
  When the service is behind/ahead it answers `409 { resync, have }` and the
  request is replayed from there. No engine indices cross the wire — only the
  (stable) setup and the coordinate action shapes.

```bash
BOT_PORT=8100 uv run --package settlrl-agents settlrl-bot-service --bot greedy
```

An **admin** registers a service by base URL at runtime (from the
[Admin](#admin) page); its bot self-identifies via `GET /info` and joins the
catalog under its own name:

| Endpoint | Description |
|---|---|
| `GET /api/admin/status` | Server status: uptime, live games, bot services (admin; see [Admin](#admin)) |
| `GET /api/admin/bot-providers` | List registered bot services (admin) |
| `POST /api/admin/bot-providers` | Register one `{ "base_url" }` (admin); `400` if unreachable |
| `DELETE /api/admin/bot-providers/{name}` | Unregister one by bot name (admin) |

`GET /api/bots` is empty until a service is registered, and every bot move is
delegated over the API. A remote service that is slow or fails (or an abandoned
human turn) falls back to a trivial local random move, so a game never stalls.
Registrations live in memory, so re-register services after a restart.

## Tests

The app builds its board coordinate tables and resource / dev-card
orderings from `settlrl-game`'s geometry and enums, and defines its own
flat action space over reference actions; the test suite checks the geometry is
well-formed, pins the enum-derived orderings, and round-trips the flat table
(every legal flat reconstructs a legal reference action and maps back to itself).

The server tests follow its layering: `test_games.py` covers the registry and
seat claims, `test_views.py` covers the per-seat snapshots —
including a sweep asserting that no observer's view ever leaks another hand —
and `test_server.py` covers only what routes own (auth, status codes, locking),
each test building its own app via `create_app`. The wire contract is pinned
twice: pytest checks the committed `frontend/openapi.json` against the live
schema, and the frontend's wire types are generated from it (`npm run gen-api`
regenerates both whenever `models.py` changes).

```bash
uv run pytest packages/settlrl-app/tests
```

A browser end-to-end suite drives the real app (create / join / spectate and
per-seat redaction over the wire); it needs a running server with a built
frontend and a system Chromium:

```bash
cd packages/settlrl-app/frontend
BASE=http://localhost:8000 npm run e2e
```

## API

| Endpoint | Description |
|---|---|
| `POST /api/games` | Create an all-bot game `{ "seed", "seats": [<bot kind>, ...] }` (2–4 seats, bots playing each other) — returns `{ "id" }`. A `"human"` seat is `422` (host a lobby instead), as is an unknown bot kind or one that doesn't support the count; `503` when every game slot is taken |
| `GET /api/games/{id}` | The requester's snapshot: board + status + their legal moves (`X-Seat-Tokens` header; omit to spectate) |
| `POST /api/games/{id}/action` | Apply the acting seat's move `{ "flat": <action index> }` — `403` without that seat's token, `409` if illegal |
| `GET /api/games/{id}/events` | Server-sent events: the requester's snapshot immediately, then again on every change (`bot_move` carries the server-paced bot play just made) |
| `POST /api/games/{id}/chat` | Append a chat message `{ "text", "player"?: <owned seat> }` (no seat: spectator) |
| `GET /api/games/{id}/record` | The finished game as a replayable `GameRecord` transcript — served for past games too, rebuilt from the store (`409` while running; `404` if unknown) |
| `POST /api/games/{id}/replay` | Load a finished game for replay — a past game too (`409` while running; `404` if unknown) |
| `POST /api/lobbies` | Open a lobby `{ "mode": "online" \| "hotseat", "seed", "n_players": 2 \| 4, "number_placement", "victory_points_to_win"?, "listed"?, "searchable"? }` — returns the lobby id and the host's seat tokens (seat 0 online; every human seat hotseat). `listed`/`searchable` require a signed-in account (`401` otherwise) |
| `GET /api/lobbies` | Open lobbies anyone can join (listed, with a free human seat), newest first (public; see [Lobby](#lobby)) |
| `GET /api/lobbies/{id}` | The requester's lobby snapshot: config, seats, chat, and the board preview rendered from the seed |
| `GET /api/lobbies/{id}/events` | Server-sent events: the requester's lobby snapshot immediately, then again on every change (joins, seat edits, config, chat), and once more carrying `started_game_id` when the game begins |
| `POST /api/lobbies/{id}/configure` | Reconfigure the lobby — any of `{ "seed"?, "n_players"?, "number_placement"?, "victory_points_to_win"?, "listed"?, "searchable"? }`, merged onto the current config. Host only (seat-0 owner, `403` otherwise); changing the player count rebuilds the seat list, keeping the host |
| `POST /api/lobbies/{id}/seats` | Host-only: retarget a seat `{ "seat", "kind": "human" \| <bot kind> }` — a bot, or human (opens it online / takes it hotseat). `409` for seat 0 or a seat another player holds; `422` for an unknown bot kind |
| `POST /api/lobbies/{id}/join` | Claim an open human seat `{ "seat"?: <n> }` (first free one by default) in an online lobby — returns the seat and its token. `409` when taken/full or the lobby isn't joinable |
| `POST /api/lobbies/{id}/leave` | Free your seats; the host (seat 0) leaving closes the whole lobby. `403` if you hold no seat here |
| `POST /api/lobbies/{id}/chat` | Append a chat message `{ "text", "player"?: <owned seat> }` — carried into the game's log on start |
| `POST /api/lobbies/{id}/start` | Host-only: build the engine and materialise the lobby into a game — returns `{ "game_id" }` (the lobby SSE carries everyone into `/play/{id}`). `409` while any human seat is still open (no bot-fill). At the concurrency cap, returns a place in line `{ "queued": true, "ticket", "position", "total" }` — re-POST with the `ticket` to keep it — and `503` when the registry is full outright |
| `GET /api/replay` | The loaded replay's opening state, or `null` when none is loaded |
| `POST /api/replay` | Load a game record (the record JSON) for replay; returns the opening state. `422` if malformed |
| `GET /api/replay/state?move=N` | The loaded replay after `N` moves (0 = the opening board). `404` until a replay is loaded |
| `GET /api/replay/record` | The loaded replay's record JSON (to save it to a file) |
| `GET /api/bots` | Bot kinds available for seats (built-in + registered remote services), each with the player counts it supports and its configurable parameters |
| `POST /api/auth/register` · `/login` · `/logout` · `GET /api/users/me` | Optional accounts (OAuth2 password flow; see [Accounts](#accounts)) |
| `GET /api/me/games` | The signed-in user's live games — seats follow the account across devices |
| `GET /api/me/history` | The signed-in user's finished games (newest first) — replayable / downloadable by id |
| `GET /api/leaderboard` | Elo ratings for accounts and bots, per player-count bucket, best first (public; see [Leaderboard](#leaderboard)) |
| `POST /api/matchmake` | Elo Quick Match `{ "n_players": 2 \| 4, "ticket"? }` — returns `{ "queued": true, "ticket", "waiting" }` to re-poll, or `{ "id", "seat", "token" }` once matched (bots fill the rest; see [Lobby](#lobby)) |
| `GET` · `POST` · `DELETE /api/admin/bot-providers` | Manage remote bot services (admin; see [Bot services](#bot-services)) |
| `GET /docs` | Interactive API docs (Swagger UI) |

Each legal move in `GET /api/games/{id}` is a decoded action descriptor carrying its `flat` index
(post it back to apply it), a `type`, a human `label`, and — depending on the type — the board
target (`vertex` / `edge` / `tile` + `victim`) or resources involved, in the same cube/axial
coordinates the board uses.

Example response:
```json
{
  "tiles": [
    { "q": 0, "r": -2, "terrain": "ore", "number": 10 },
    { "q": 1, "r": -2, "terrain": "sheep", "number": 2 },
    ...
  ]
}
```

Tile position uses **axial coordinates** with a pointy-top hex orientation. The board is a hexagon of radius 2 (19 tiles) centred on `(0, 0)`. The layout (terrain and number tokens) is generated by `settlrl-game`'s reference rules, so it is randomised per server start rather than fixed.

## Project layout

```
packages/settlrl-app/
├── src/settlrl_app/      # grouped by layer; server.py wires them together
│   ├── __init__.py      # CLI entry point (uvicorn)
│   ├── config.py        # Settings (pydantic-settings): the env-var config, typed in one place
│   ├── server.py        # create_app composition root: wires the app, mounts routers + SPA
│   ├── api/             # the HTTP layer (game model + serialization is settlrl-game)
│   │   ├── deps.py        # Shared request helpers + the runtime context (Deps) routers close over
│   │   ├── routers/       # Routes by area: games, lobbies, quickmatch, replay, bots, me, leaderboard, admin (each build(deps) -> APIRouter)
│   │   ├── views.py       # Per-seat snapshots: the hidden-information boundary
│   │   └── openapi.py     # Schema dump backing the generated frontend types
│   ├── game/            # the live game runtime
│   │   ├── games.py       # Game registry: ids, per-game locks, seat claims (tokens), the create queue
│   │   ├── lobbies.py     # Pre-game lobbies: config + membership staging, materialised into a game on start
│   │   ├── driver.py      # Per-game asyncio task: bot pacing (remote) + idle-turn timeouts
│   │   ├── matchmaking.py  # Elo Quick Match: pool waiters per count, pair + bot-fill the table
│   │   └── replay.py      # ReplaySession: a loaded record replayed into per-move snapshots
│   ├── bots/            # the bot seam (no agent code runs here)
│   │   └── providers.py   # Bot kinds -> the registered remote services that serve them
│   ├── ratings.py       # Multiplayer rating via openskill: winner-takes-all skill updates
│   └── storage/         # the one async DB: identity + persistence
│       ├── db.py          # The async SQLAlchemy engine: users, login tokens, game journals, ratings
│       ├── auth.py        # Optional accounts: fastapi-users (DatabaseStrategy) on the shared DB
│       └── store.py       # Crash-recovery journals + skill ratings on the shared DB (write-behind)
├── tests/               # Pytest: board conversion, flat-table round-trip, per-seat views, server
└── frontend/
    ├── openapi.json     # Committed wire schema (pinned by pytest; npm run gen-api)
    ├── e2e/             # Browser end-to-end checks (npm run e2e)
    └── src/
        ├── App.tsx          # Routes: menu, /play (→ /lobby), /play/:id, /lobby, /lobby/:id, /login, /register, /help, /profile, /leaderboard, /replay, /admin
        ├── lib/hex.ts        # Axial/cube → pixel conversion, hex corner math, coord equality
        ├── lib/api.ts        # JSON fetch wrapper (ApiError) + the SSE reader
        ├── lib/client.ts     # Typed REST client (openapi-fetch) from the schema, auth-injecting
        ├── lib/queries.ts    # React Query hooks for the read endpoints (me/games, history, leaderboard, lobby)
        ├── lib/boardData.ts  # Board types + palette + resource/card constants + adaptBoard
        ├── lib/api-schema.d.ts # Wire types generated from openapi.json (do not edit)
        ├── lib/game.ts       # Live-game API client (/api/game*)
        ├── lib/lobby.ts      # Lobby API client (/api/lobbies*)
        ├── lib/useLobby.ts   # Hook driving one lobby (snapshot over its SSE stream, chat)
        ├── lib/transfers.ts  # Diff two snapshots into card-transfer animations (production / steals)
        ├── lib/replay.ts     # Replay API client (/api/replay*)
        ├── lib/actionMeta.ts # Action display metadata: icons, labels, costs, confirm phrasing
        ├── lib/useGame.ts    # Hook driving one live game (snapshot stream, act / chat)
        ├── lib/seats.ts      # Seat tokens this browser holds, per game (localStorage)
        ├── lib/auth.ts       # Account / auth client (/api/auth*): bearer token in localStorage
        ├── lib/admin.ts      # Admin client (/api/admin/*): the bot-service registrations
        ├── lib/clientId.ts   # Stable anonymous browser id (the X-Client-Id guest guard)
        ├── lib/viewport.ts   # useTableViewport: pan / zoom / rotate (mouse, touch, keyboard)
        ├── lib/theme.ts      # Light / dark theme switching (persisted)
        ├── lib/sound.ts      # Action sound effects, derived from new game-log entries
        ├── lib/cx.ts         # Class-name joiner for conditional CSS-module classes
        ├── lib/ui.ts         # The few board-layer style helpers the SVG renderers still need
        ├── index.css         # Design tokens (theme colours + spacing/radii/type) + keyframes
        ├── styles/ui.module.css # Shared CSS-module classes (panel, button, page/toolbar, …)
        ├── components/Button.tsx · Panel.tsx # Primitives over styles/ui.module.css
        ├── pages/
        │   ├── Menu.tsx       # Landing page: choose Play (the lobby), Replay, or Leaderboard
        │   ├── PlayView.tsx   # Play mode: game state + handlers wiring the components below
        │   ├── LobbyView.tsx  # Lobby hub: host a game, Quick Match, or join an open lobby
        │   ├── LobbyRoom.tsx  # A lobby's staging room (/lobby/:id): players / map+settings / chat
        │   ├── AuthView.tsx   # Sign-in / registration page (/login, /register)
        │   ├── ProfileView.tsx # Profile: live games + finished-game history
        │   ├── LeaderboardView.tsx # Per-player-count Elo ladders
        │   ├── AdminView.tsx  # Superuser status page + bot-service management
        │   ├── HelpView.tsx   # Help page: controls, action icons, seats
        │   └── ReplayView.tsx # Replay mode: load a record, scrub / step / play it
        └── components/
            ├── TopBar.tsx       # Back-to-menu + mode label + theme toggle + view actions
            ├── AccountMenu.tsx  # Sign-in link / signed-in account menu (top bar)
            ├── BoardView.tsx    # The table scene: composes everything below in one SVG
            ├── InteractionOverlay.tsx # Legal-placement markers / hover ghosts / robber tiles
            ├── BoardPopover.tsx # Anchored action chooser (confirm + cost / victim pick)
            ├── ChoicePopover.tsx # Bottom-panel resource picker (monopoly / plenty)
            ├── MaritimePopover.tsx # Bank-pile picker: which resource to give + the rate
            ├── TradePopover.tsx # Compose a 1:1 player trade offer
            ├── TradeResponsePopover.tsx # Accept / Reject card for an incoming offer
            ├── Anchored.tsx     # Floating-UI positioning wrapper the popovers share
            ├── Modal.tsx        # Focus-trapped modal dialog (Radix)
            ├── GameOverScreen.tsx # End-of-game overlay: winner + final standings
            ├── MyGames.tsx      # The signed-in user's live games list
            ├── AdminLink.tsx    # Superuser-only corner link to /admin
            ├── Hand.tsx         # The acting human's chips (resources, dev cards; clickable)
            ├── CountBadge.tsx   # Cream count badge for chip corners (matches CardPile's token)
            ├── CardPile.tsx     # Top-down card pile + count token (bank, player decks)
            ├── BankStacks.tsx   # The bank's card grid (resource piles + dev deck)
            ├── PlayerAreas.tsx  # Each seat's table edge: hand/dev piles + unbuilt pieces
            ├── TableDice.tsx    # The dice on the table (click to roll when glowing)
            ├── TransferAnimations.tsx # Chips that fly between bank piles / seats on a transfer
            ├── PlayersPanel.tsx # Seat list atop the chat column (stats + belief inspect)
            ├── ChatPanel.tsx    # Right-hand column: players section + chat / log
            ├── ThemeToggle.tsx  # Light / dark switch
            ├── SoundToggle.tsx  # Mute / unmute the action sounds
            ├── icons.tsx        # Monochrome line icons (currentColor, theme-following)
            ├── HexTile.tsx      # Hex polygon, terrain colour, icon-and-number token
            ├── TerrainIcon.tsx  # Per-terrain silhouette motif (pine, sheep, …)
            ├── ResourceGlyph.tsx # A resource's terrain motif as a standalone inline glyph
            ├── Road.tsx         # Player road along an edge
            ├── Building.tsx     # Settlement / city on a vertex
            ├── Robber.tsx       # Robber pawn on a tile
            └── Port.tsx         # Harbour badge (2:1 / 3:1) with docks
```
