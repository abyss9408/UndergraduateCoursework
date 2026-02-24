# Asteroids Multiplayer — CSD2161 Assignment 4

A networked, real-time multiplayer Asteroids game built with a custom UDP client-server architecture over LAN. Supports 2–4 concurrent players with an authoritative server, lockstep asteroid destruction, and full single-player mode preservation.

---

## Table of Contents

- [Projects](#projects)
- [Quick Start](#quick-start)
- [Controls](#controls)
- [Configuration](#configuration)
- [Network Protocol](#network-protocol)
- [Architecture](#architecture)
- [Lockstep ACK System](#lockstep-ack-system)
- [Build Requirements](#build-requirements)

---

## Projects

The solution (`CSD2161_Asteroids_Multiplayer.sln`) contains two independent projects:

| Project | Description |
|---------|-------------|
| `CSD2161_Asteroids_Multiplayer` | Game client — AlphaEngine window, single-player and multiplayer modes |
| `AsteroidsServer` | Standalone UDP server — no graphics dependency, runs as a console app |

Compiled binaries are output to `Bin/`.

---

## Quick Start

### 1. Build (Visual Studio 2022, x64 Debug or Release)

Build both projects from the solution. Start the server first.

### 2. Configure `network.cfg`

Edit `CSD2161_Asteroids_Multiplayer/network.cfg` on each client machine:

```ini
server_ip=127.0.0.1    # IP address of the machine running AsteroidsServer
server_port=9999       # Must match the server's port
player_name=Player1    # Display name (max 16 characters)
min_players=2          # Game starts once this many players connect
max_players=4          # Server rejects connections beyond this
```

### 3. Run

1. Launch `AsteroidsServerD.exe` (or the Release build) on the host machine.
2. Launch `CSD2161_Asteroids_MultiplierD.exe` on each player's machine.
3. Press **2** at the main menu to join the lobby.
4. The game starts automatically once `min_players` have connected.

For a local test, run the server and both clients on the same machine with `server_ip=127.0.0.1`.

---

## Controls

| Key | Action |
|-----|--------|
| `Up Arrow` | Accelerate forward |
| `Down Arrow` | Decelerate / reverse thrust |
| `Left Arrow` | Rotate counter-clockwise |
| `Right Arrow` | Rotate clockwise |
| `Space` | Fire (3-bullet spread) |
| `ESC` | Return to menu / quit lobby |

**Main Menu**

| Key | Action |
|-----|--------|
| `1` | Single-player mode |
| `2` | Multiplayer lobby |
| `ESC` | Quit |

---

## Configuration

`network.cfg` is parsed by both the client (`ConfigReader::Load`) and, optionally, the server (`ServerMain`). Lines beginning with `#` and blank lines are ignored. Whitespace around `=` is trimmed.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `server_ip` | string | `127.0.0.1` | Server hostname or IPv4 address |
| `server_port` | uint16 | `9999` | UDP port to bind/connect |
| `player_name` | string | `Player1` | In-game display name (truncated to 16 chars) |
| `min_players` | int | `2` | Minimum players before game auto-starts |
| `max_players` | int | `4` | Maximum simultaneous players (1–4) |

---

## Network Protocol

All packets share an 8-byte packed header followed by a variable-length payload.

### Wire Header (`#pragma pack(push, 1)`)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 1 | `msgType` | `MessageType` enum value |
| 1 | 1 | `senderId` | `0x00`–`0x03` = players; `0xFF` = server |
| 2 | 2 | `seqNum` | Monotonically increasing sequence number |
| 4 | 2 | `payloadLen` | Payload byte count |
| 6 | 2 | `checksum` | XOR of header bytes 0–5 |

### Message Types

**Connection**

| Type | Code | Dir | Payload |
|------|------|-----|---------|
| `MSG_CONNECT_REQUEST` | `0x01` | C→S | `name[16]` |
| `MSG_CONNECT_ACCEPT` | `0x02` | S→C | `assignedId(1)`, `playerCount(1)`, `names[4][16]` |
| `MSG_CONNECT_REJECT` | `0x03` | S→C | `reason(1)` |
| `MSG_DISCONNECT` | `0x04` | Both | `playerId(1)`, `reason(1)` |
| `MSG_HEARTBEAT` | `0x05` | C→S | _(empty)_ |
| `MSG_HEARTBEAT_ACK` | `0x06` | S→C | _(empty)_ |

**Game Flow**

| Type | Code | Dir | Payload |
|------|------|-----|---------|
| `MSG_GAME_START` | `0x10` | S→C | `rngSeed(4)` |
| `MSG_GAME_STATE_SYNC` | `0x11` | S→C | Full snapshot (reconnect) |
| `MSG_GAME_OVER` | `0x12` | S→C | `winnerId(1)`, `scores[4](16)`, `top5(100)` |

**Real-Time**

| Type | Code | Dir | Payload |
|------|------|-----|---------|
| `MSG_PLAYER_INPUT` | `0x20` | C→S | `inputBits(1)`, `dt(4)` |
| `MSG_SHIP_STATE` | `0x21` | S→C | `playerId(1)`, `posX(4)`, `posY(4)`, `velX(4)`, `velY(4)`, `dir(4)` |
| `MSG_BULLET_SPAWN` | `0x22` | S→C | `bulletId(2)`, `ownerId(1)`, `pos(8)`, `vel(8)`, `dir(4)` |
| `MSG_BULLET_DESTROY` | `0x23` | S→C | `bulletId(2)` |

**Asteroids (lockstep)**

| Type | Code | Dir | Payload |
|------|------|-----|---------|
| `MSG_ASTEROID_SPAWN` | `0x30` | S→C | `id(2)`, `pos(8)`, `vel(8)`, `scale(8)` |
| `MSG_ASTEROID_CORRECT` | `0x31` | S→C | `id(2)`, `pos(8)`, `vel(8)` |
| `MSG_ASTEROID_HIT` | `0x32` | C→S | `asteroidId(2)`, `bulletId(2)` |
| `MSG_ASTEROID_DESTROY` | `0x33` | S→C | `id(2)`, `scoringPlayer(1)`, `spawnCount(1)` |
| `MSG_ASTEROID_DESTROY_ACK` | `0x34` | C→S | `asteroidId(2)` |
| `MSG_ASTEROID_DESTROY_CONFIRM` | `0x35` | S→C | `asteroidId(2)` |

**Score**

| Type | Code | Dir | Payload |
|------|------|-----|---------|
| `MSG_SCORE_UPDATE` | `0x40` | S→C | `playerId(1)`, `score(4)` |

### Input Bit Flags

```
0x01  INPUT_UP      Forward thrust
0x02  INPUT_DOWN    Reverse thrust
0x04  INPUT_LEFT    Rotate CCW
0x08  INPUT_RIGHT   Rotate CW
0x10  INPUT_SHOOT   Fire
```

---

## Architecture

### Server — 3 threads

```
[Receive Thread]  recvfrom() loop → validate checksum → push InPacket to _inboundQueue
[Game Loop Thread] 60 Hz tick:
    ProcessInboundMessages()     – drain _inboundQueue, dispatch handlers
    TickShips / Bullets / Asteroids(dt)
    CheckCollisions()
    CheckPendingAsteroidAcks()
    BroadcastShipStates()        – every 2 ticks  (~33 ms)
    BroadcastAsteroidCorrections() – every 10 ticks (~166 ms)
    CheckHeartbeatTimeouts()
[Send Thread]     drain _outboundQueue → sendto()
```

### Client — 2 threads

```
[Main / Render Thread]  AlphaEngine game loop:
    applyNetworkState()   – drain _inboundQueue, call MP_*() helpers
    collect inputBits → onLocalInput() → sendto server
    client-side prediction for local ship
    draw all active game object instances
[Receive Thread]  recvfrom() loop → validate → push to _inboundQueue
    immediately sends MSG_ASTEROID_DESTROY_ACK on receipt (no main-thread wait)
```

### Server game object limits

| Object | Max |
|--------|-----|
| Players | 4 |
| Asteroids | 256 |
| Bullets | 512 |

### Game constants

| Constant | Value |
|----------|-------|
| Win score (MP) | 5 000 pts |
| Win score (SP) | 10 000 pts |
| Heartbeat timeout | 3 s |
| Play area | ±400 × ±300 units |

---

## Lockstep ACK System

Asteroid destruction is the one operation that must be fully consistent across all clients. The protocol ensures no client removes an asteroid before all have acknowledged it.

```
Client detects hit → MSG_ASTEROID_HIT ──────────────────→ Server
                                                           Server validates
                                                           Awards score
Server ←── MSG_ASTEROID_DESTROY (id, scoringPlayer) ─────→ All clients
                                                           Clients: mark dying, play explosion
                                                           Recv thread sends ACK immediately:
Each client ──── MSG_ASTEROID_DESTROY_ACK ───────────────→ Server
                                                           Server collects ackBits
                                                           (force-sets after 500 ms timeout)
Server ←── MSG_ASTEROID_DESTROY_CONFIRM (id) ────────────→ All clients
                                                           Clients: call gameObjInstDestroy()
```

**Retry policy:** Server retransmits `MSG_ASTEROID_DESTROY` every 200 ms, up to 3 times, to players that have not yet ACK'd. After 500 ms total, any missing ACK is force-accepted so a single slow client cannot freeze the game.

---

## Build Requirements

- **OS:** Windows 10/11 (Winsock2)
- **IDE:** Visual Studio 2022 (toolset v143)
- **Standard:** C++20 (`stdcpp20`)
- **Client dependencies:** AlphaEngine (included in `Dep/`), `ws2_32.lib`
- **Server dependencies:** `ws2_32.lib` only (no graphics)

Build configurations available: `Debug|x64`, `Release|x64`, `Debug|Win32`, `Release|Win32`.

> **Include order note:** Any `.cpp` that uses both `<winsock2.h>` (via `MultiplayerMode.h` or `NetworkMessage.h`) and `<windows.h>` (via `AEEngine.h`) must include the winsock header first. All affected files already do this.
