# Spaceships - Multiplayer Asteroids Game

## Overview

Spaceships is a multiplayer adaptation of our previous module's assignment. Players control spaceships and navigate through a field of asteroids, shooting at them to score points while avoiding collisions. The game supports both single-player mode and multiplayer with 4 players over a local area network (LAN).

## Running the Game

The game has three modes of operation:

### 1. Single Player Mode

- Launch the game executable `CSD2161_Asteroids_Multiplayer.exe` inside `Bin` folder
- When prompted for game mode, select **"YES"** for Single Player mode
- Game will start immediately

### 2. Multiplayer Server Mode

- Launch the game executable `CSD2161_Asteroids_Multiplayer.exe` inside `Bin` folder
- When prompted for game mode, select **"CANCEL"** for Multiplayer Server mode
- This will launch a separate server application (`AsteroidsServer.exe`)
- The server will display its IP address and port (27015)
- The server will automatically start the game when enough clients connect
- Press 'Q' to shut down the server

### 3. Multiplayer Client Mode

- Launch the game executable `CSD2161_Asteroids_Multiplayer.exe` inside `Bin` folder
- When prompted for game mode, select **"NO"** for Multiplayer Client mode
- Enter the server's IP address when prompted
  - The address will be saved in `network.cfg` inside `Bin` folder for future use
  - By default, it will suggest `127.0.0.1` (localhost)
- The game will connect to the server and start when enough players have joined

## Game Controls

- **Arrow Up**: Accelerate forward
- **Arrow Down**: Decelerate/move backward
- **Arrow Left**: Rotate counterclockwise
- **Arrow Right**: Rotate clockwise
- **Spacebar**: Fire weapons (shoot three bullets)
- **Escape**: Exit the game

## Gameplay Rules

- Each player controls a spaceship
- Destroy asteroids by shooting them to earn 100 points per asteroid
- Avoid collisions with asteroids - collisions cost you one life
- The first player to reach 5,000 points wins
- If all players except one lose all their lives, the remaining player wins
- The game automatically restarts after showing the winner

## Configuration

### Server Configuration

The server runs on port 27015 by default. This port needs to be available and not blocked by a firewall.

### Client Configuration

The client stores the last used server IP address in the `network.cfg` file inside `Bin` folder, which is created automatically when you first connect to a server. The file format is:

```
ServerAddress = 192.168.1.100
```

You can manually edit this file to change the default server address.

## Troubleshooting

### Connection Issues

1. **Cannot connect to server**:
   - Make sure the server is running
   - Check that the IP address is correct
   - Verify that port 27015 is not blocked by a firewall

2. **Game crashes on startup**:
   - Make sure all necessary DLL files are in the same directory as the executable

3. **Players can't see each other**:
   - Make sure all players are on the same local network
   - Check that the server IP address is reachable from all clients

### Performance Issues

1. **Game lag**:
   - Reduce the number of objects on screen (fewer players)
   - Make sure the network has sufficient bandwidth

## Development Notes

The game uses UDP for networking, which is optimized for real-time gameplay but may occasionally result in packet loss. The game implements various synchronization techniques to maintain consistency across clients:

- Object creation/destruction acknowledgments
- Regular state updates (20 per second)
- Client-side prediction and interpolation

If you encounter bugs or have suggestions, please report them to the development team.