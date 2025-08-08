<template>
  <div class="game-container">
    <h2>Hand Tracking Game</h2>
    <p v-if="loading">Loading video feed...</p>
    
    <div v-if="!gameStarted" class="welcome-screen">
      <div class="welcome-card">
        <h3>Welcome!</h3>
        <p>Catch the falling balls with your hands.</p>
        <p>Make sure your webcam is enabled.</p>
        <button @click="startGame">Start Game</button>
      </div>
    </div>

    <div v-else class="game-display">
      <img :src="videoSrc" alt="Video Feed" class="video-feed" @load="loading = false" @error="handleVideoError">
      <div class="overlay">
        <div class="score-display">
          <!-- <p>Score: {{ score }}</p> -->
        </div>
      </div>

      <div v-if="isGameOver" class="game-over-message">
        <p>Game Over!</p>
        <button @click="restartGame">Play Again</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const score = ref(0);
const isGameOver = ref(false);
const gameStarted = ref(false);
const loading = ref(true);
const videoSrc = ref('');
let scoreInterval = null;

// Function to fetch the score and game over status from the Flask backend
async function fetchScore() {
  try {
    const response = await fetch('http://127.0.0.1:5000/score');
    if (!response.ok) {
      throw new Error('Failed to fetch score');
    }
    const data = await response.json();
    score.value = data.score;
    isGameOver.value = data.game_over;
  } catch (error) {
    console.error('Error fetching score:', error);
  }
}

// Function to start the game
function startGame() {
  gameStarted.value = true;
  loading.value = true;
  videoSrc.value = 'http://127.0.0.1:5000/video_feed';
  
  // Fetch the score every second to update the display
  scoreInterval = setInterval(fetchScore, 1000);
}

// Function to restart the game
async function restartGame() {
  try {
    const response = await fetch('http://127.0.0.1:5000/restart', { method: 'POST' });
    if (response.ok) {
      // Clear the game over state and start over
      isGameOver.value = false;
      score.value = 0;
      startGame();
    }
  } catch (error) {
    console.error('Error restarting game:', error);
  }
}

function handleVideoError() {
  console.error("Error loading video feed. Make sure the Flask server is running.");
  loading.value = false;
  // You could display a user-friendly error message here
}

onMounted(() => {
  // Clean up the interval when the component is unmounted
  // (e.g., if the user navigates away)
  window.addEventListener('beforeunload', () => {
    clearInterval(scoreInterval);
  });
});
</script>

<style scoped>
.game-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.welcome-screen, .game-over-message {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: rgba(0, 0, 0, 0.8);
  padding: 30px 40px;
  border-radius: 15px;
  z-index: 10;
  border: 2px solid #00ffff;
  box-shadow: 0 0 15px #00ffff;
}
.welcome-screen h3 {
  color: #00ffff;
  text-shadow: 0 0 5px #00ffff;
}
.welcome-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
}
.game-over-message p {
  color: #ff4d4d;
  font-size: 2em;
  font-weight: bold;
  text-shadow: 0 0 10px #ff4d4d;
  margin-bottom: 20px;
}
button {
  background: #00ffff;
  color: #1a1a2e;
  border: none;
  padding: 10px 20px;
  border-radius: 5px;
  font-family: 'Orbitron', sans-serif;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
}
button:hover {
  background: #fff;
  box-shadow: 0 0 20px rgba(0, 255, 255, 1);
}
/* Default (Mobile - Small Screens) */
.game-display {
  width: 100%;
  height: 300px; /* Smaller height for mobile */
  max-width: 100%;
  background-color: #000;
  border: 3px solid #fff; /* Thinner border for small screens */
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
  margin: 0 auto;
}

/* Medium Screens (Tablets, 600px - 992px) */
@media (min-width: 600px) {
  .game-display {
    height: 450px;
    border-width: 4px;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.5);
  }
}

/* Large Screens (Desktops, 992px and above) */
@media (min-width: 992px) {
  .game-display {
    width: 1200px;
    height: 600px; /* Original size */
    border-width: 5px;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
  }
}
.video-feed {
  width: 100%;
  height: 100%;
  border-radius: 5px;
}
.overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none; /* Allows clicks to pass through */
}
.score-display {
  position: absolute;
  top: 10px;
  left: 10px;
  font-size: 1.5em;
  text-shadow: 2px 2px 4px #000;
  color: #00ffff;
}
</style>