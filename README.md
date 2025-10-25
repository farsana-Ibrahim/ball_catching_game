Hand Tracking Ball Catching Game - README

A fun and interactive hand-tracking game where players catch falling balls using their hands detected through a webcam. Built with Flask backend and modern frontend.

🎮 Features

- Real-time Hand Tracking: Uses MediaPipe for accurate hand detection
- Gesture Control: Control the basket with your hand movements
- Progressive Difficulty: Levels up as you score more points
- Special Balls: Golden balls worth extra points
- Visual Effects: Particle effects, animations, and trophy display
- Sound Effects: Background music and game sounds
- Responsive Design: Clean, space-themed UI

🛠️ Tech Stack

 Backend
- Flask - Web framework
- OpenCV - Computer vision and video processing
- MediaPipe - Hand tracking and gesture recognition
- PyGame - Audio handling
- PIL (Pillow) - Image processing

 Frontend
- HTML5/CSS3 - Structure and styling
- JavaScript (ES6+) - Game logic and API integration
- Orbitron Font - Futuristic typography

📋 Prerequisites

- Python 3.7+
- Webcam
- Modern web browser with camera access

 🚀 Installation

 Backend Setup

1. Clone the repository
   ```
   git clone <your-repo-url>
   cd hand-tracking-game
   ```

2. Create virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install flask opencv-python mediapipe numpy pygame pillow
   ```

4. Prepare sound files
   - Create a `static` folder in the backend directory
   - Add these sound files:
     - `catch.wav` - Sound when catching a ball
     - `congrats.wav` - Game completion sound
     - `happy_music.wav` - Background music

5. Run the backend
   ```
   python app.py
   ```
   The server will start at `http://localhost:5000`

Frontend Setup

1. Serve the frontend
   - Use any static file server or open `index.html` directly in a browser
   - For development, you can use:
     ```
     python -m http.server 8000
     ```

 🎯 How to Play

1. Start the Game: Open the frontend in your browser
2. Allow Camera Access: Grant permission when prompted
3. Position Your Hands: Place your hands in front of the camera
4. Catch Balls: Move your hands to position the basket under falling balls
5. Score Points: 
   - Regular balls: 1 point
   - Golden balls: 5 points
6. Level Up: Every 10 points increases the game level and ball speed
7. Game Duration: 30 seconds per round

 🎮 Controls

- Hand Movement: Move your hands left/right to control the basket
- Automatic Tracking: The game automatically detects your index fingers or palms
- Space Bar: Press space to restart after game over

 📁 Project Structure

```
hand-tracking-game/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── static/
│   │   ├── catch.wav          # Catch sound effect
│   │   ├── congrats.wav       # Victory sound
│   │   └── happy_music.wav    # Background music
│   └── trophy.jpg            # Trophy image (optional)
├── frontend/
│   ├── index.html            # Main HTML file
│   ├── src/
│   │   └── main.js           # Frontend JavaScript
│   └── styles/               # CSS stylesheets
└── README.md
```

 🔧 API Endpoints

- `GET /video_feed` - Live video stream with game overlay
- `GET /score` - Get current score and game status
- `POST /restart` - Restart the game

 ⚙️ Configuration

Game Parameters (Modifiable in backend/app.py)
```python
SCREEN_WIDTH = 640           # Game screen width
SCREEN_HEIGHT = 480          # Game screen height
BASKET_HEIGHT = 100          # Basket size
BALL_RADIUS = 20             # Ball size
BALL_SPEED = 4               # Initial ball speed
GAME_DURATION = 30           # Game duration in seconds
LEVEL_UP_SCORE = 10          # Points needed to level up
```

 🐛 Troubleshooting

 Common Issues

1. Webcam not detected
   - Check if another application is using the camera
   - Verify camera permissions

2. Sound files not found
   - Ensure sound files are in the correct `static` folder
   - Check file paths in the code

3. Hand tracking not working
   - Ensure good lighting
   - Keep hands clearly visible to camera
   - Avoid busy backgrounds

4. Performance issues
   - Close other applications using the camera
   - Reduce game resolution in code

🚀 Future Enhancements

- [ ] Multiplayer support
- [ ] Different game modes
- [ ] Power-ups and special abilities
- [ ] Leaderboard system
- [ ] Mobile app version
- [ ] Additional gesture controls


 🙏 Acknowledgments

- MediaPipe for hand tracking technology
- OpenCV community for computer vision tools
- Flask framework for backend infrastructure

---

**Note: This game requires a webcam and works best in well-lit environments. For optimal performance, ensure your hands are clearly visible to the camera.
