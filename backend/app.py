from flask import Flask, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import pygame
import sys
from PIL import Image, ImageDraw

app = Flask(__name__)

# Initialize pygame mixer for sound
pygame.mixer.init()
try:
    catch_sound = pygame.mixer.Sound("C:/Users/Farsana Ibrahim/Desktop/game_prjct/backend/static/catch.wav")
    game_over_sound = pygame.mixer.Sound("C:/Users/Farsana Ibrahim/Desktop/game_prjct/backend/static/congrats.wav")
    background_music = pygame.mixer.Sound("C:/Users/Farsana Ibrahim/Desktop/game_prjct/backend/static/happy_music.wav")
    background_music.play(-1)  # Loop indefinitely
    print("DEBUG: Sound files loaded successfully.")
except Exception as e:
    print(f"DEBUG: Sound files not found or error loading: {e}. Game will run without sound.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    static_image_mode=False
)
print("DEBUG: MediaPipe Hands initialized.")

# Game parameters
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BASKET_HEIGHT = 100
BASKET_INITIAL_WIDTH = 80
BALL_RADIUS = 20
BALL_SPEED = 4
SCORE = 0
GAME_DURATION = 30
LEVEL_UP_SCORE = 10

# Hand tracking improvements
SMOOTHING_FACTOR = 0.7
prev_x = SCREEN_WIDTH // 2
prev_y = SCREEN_HEIGHT - 100
prev_width = BASKET_INITIAL_WIDTH

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open video capture device (webcam).")
else:
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    print("DEBUG: OpenCV video capture initialized.")

# Ball class (unchanged)
class Ball:
    def __init__(self, level):
        self.x = random.randint(BALL_RADIUS, SCREEN_WIDTH - BALL_RADIUS)
        self.y = -BALL_RADIUS
        self.speed = BALL_SPEED + random.random() * 2 + (level * 0.2)
        self.type = random.randint(0, 3)
        self.is_special = random.random() < 0.1
        if self.is_special:
            self.color = (255, 215, 0)
            self.points = 5
            self.size = BALL_RADIUS * 1.5
        else:
            self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            self.points = 1
            self.size = BALL_RADIUS

    def update(self):
        self.y += self.speed

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), int(self.size), self.color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), int(self.size), (0, 0, 0), 2)
        cv2.circle(frame, (int(self.x)-int(self.size/3), int(self.y)-int(self.size/3)), int(self.size/6), (0, 0, 0), -1)
        cv2.circle(frame, (int(self.x)+int(self.size/3), int(self.y)-int(self.size/3)), int(self.size/6), (0, 0, 0), -1)
        cv2.ellipse(frame, (int(self.x), int(self.y)+int(self.size/4)), (int(self.size/3), int(self.size/5)), 0, 0, 180, (0, 0, 0), 2)

# Basket class (unchanged)
class Basket:
    def __init__(self, initial_width, initial_height):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT - 100
        self.width = initial_width
        self.height = initial_height
        self.particles = []
        self.basket_img = None
        self._render_basket_image()

    def update_position(self, target_x, target_y, target_width):
        self.x = int(self.x * SMOOTHING_FACTOR + target_x * (1 - SMOOTHING_FACTOR))
        self.y = int(self.y * SMOOTHING_FACTOR + target_y * (1 - SMOOTHING_FACTOR))
        new_width = int(self.width * SMOOTHING_FACTOR + target_width * (1 - SMOOTHING_FACTOR))
        if new_width != self.width:
            self.width = new_width
            self._render_basket_image()
        self.x = max(self.width // 2, min(self.x, SCREEN_WIDTH - self.width // 2))
        self.y = max(self.height // 2, min(self.y, SCREEN_HEIGHT - self.height // 2))
        self.width = max(BASKET_INITIAL_WIDTH, min(self.width, BASKET_INITIAL_WIDTH))

    def _render_basket_image(self):
        self.basket_img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(self.basket_img)
        basket_color = (139, 69, 19)
        handle_color = (160, 82, 45)
        y_top_pattern = 20
        y_bottom_pattern = max(y_top_pattern + 1, self.height - 10)
        for i in range(0, self.width, 10):
            draw.rectangle([(i, y_top_pattern), (i+5, y_bottom_pattern)], fill=basket_color)
        for i in range(y_top_pattern, y_bottom_pattern, 10):
            draw.rectangle([(0, i), (self.width, i+5)], fill=basket_color)
        draw.ellipse([(10, 5), (self.width-10, 25)], outline=handle_color, width=3)
        draw.arc([(self.width//2-30, -15), (self.width//2+30, 25)], start=180, end=360, fill=handle_color, width=5)

    def draw(self, frame):
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x1 = int(self.x - self.width // 2)
        y1 = int(self.y - self.height // 2)
        x1 = max(0, min(x1, SCREEN_WIDTH - self.width))
        y1 = max(0, min(y1, SCREEN_HEIGHT - self.height))
        if self.basket_img.width != self.width or self.basket_img.height != self.height:
            resized_basket = self.basket_img.resize((self.width, self.height), Image.LANCZOS)
        else:
            resized_basket = self.basket_img
        frame_pil.paste(resized_basket, (x1, y1), resized_basket)
        frame[:] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        for particle in self.particles[:]:
            particle[0] += particle[2]
            particle[1] += particle[3]
            particle[4] -= 1
            if particle[4] <= 0:
                self.particles.remove(particle)
            else:
                cv2.circle(frame, (int(particle[0]), int(particle[1])), int(particle[4]/2), (255, 255, 255), -1)

    def add_particles(self, count=15):
        for _ in range(count):
            self.particles.append([
                self.x + random.randint(-self.width//2, self.width//2),
                self.y + random.randint(-self.height//2, self.height//2),
                random.uniform(-3, 3),
                random.uniform(-5, -1),
                random.randint(15, 30)
            ])

# Trophy class (unchanged)
class Trophy:
    def __init__(self):
        try:
            self.img = Image.open("trophy.jpg").convert("RGBA")
            self.img = self.img.resize((150, 250), Image.LANCZOS)
            self.alpha = 0
            self.fade_in = True
            print("DEBUG: Trophy image loaded successfully.")
        except Exception as e:
            print(f"ERROR: Could not load trophy image: {e}.")
            self.img = None
            self.alpha = 0
            self.fade_in = False

    def update(self):
        if self.img is not None and self.fade_in and self.alpha < 1:
            self.alpha += 0.02
        elif self.img is not None and self.alpha >= 1:
            self.fade_in = False

    def draw(self, frame):
        if self.img is None:
            return
        try:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = SCREEN_WIDTH - 180
            y = SCREEN_HEIGHT // 2 - 75
            mask = self.img.split()[3]
            mask = mask.point(lambda p: p * self.alpha)
            frame_pil.paste(self.img, (x, y), mask)
            frame[:] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"ERROR: Error drawing trophy: {e}")

# Game state global variables
balls = []
basket = Basket(BASKET_INITIAL_WIDTH, BASKET_HEIGHT)
trophy = Trophy()
start_time = time.time()
game_over = False
sound_played = False
level = 1
stars = [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT),
          random.randint(1, 3), random.random()) for _ in range(50)]

def generate_frames():
    global balls, basket, trophy, start_time, game_over, sound_played, level, SCORE, prev_x, prev_y, prev_width

    while True:
        current_time = time.time()
        remaining_time = max(0, GAME_DURATION - (current_time - start_time))
        level = 1 + SCORE // LEVEL_UP_SCORE

        if len(balls) == 0 and remaining_time > 0:
            balls.append(Ball(level))

        success, frame = cap.read()
        if not success:
            print("--- DEBUG: FAILED TO READ FRAME FROM CAMERA! ---")
            time.sleep(1)
            continue

        if frame is None:
            print("--- DEBUG: RECEIVED NONE FRAME FROM CAMERA! ---")
            time.sleep(1)
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand with MediaPipe
        results = hands.process(rgb_frame)

        left_hand_x = None
        right_hand_x = None
        main_hand_y = None

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if results.multi_handedness and results.multi_handedness[i].classification:
                    handedness = results.multi_handedness[i].classification[0].label
                else:
                    handedness = 'Unknown'

                # Get FINGERTIP (INDEX_FINGER_TIP)
                fingertip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                fingertip_x = int(fingertip_landmark.x * SCREEN_WIDTH)
                fingertip_y = int(fingertip_landmark.y * SCREEN_HEIGHT)

                # Get PALM (WRIST)
                wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                palm_x = int(wrist_landmark.x * SCREEN_WIDTH)
                palm_y = int(wrist_landmark.y * SCREEN_HEIGHT)

                # Decide whether to use FINGERTIP or PALM
                use_fingertip = True  # Default to fingertip
                # If fingertip is near screen edges, use palm instead
                if (fingertip_x < 50 or fingertip_x > SCREEN_WIDTH - 50 or
                    fingertip_y < 50 or fingertip_y > SCREEN_HEIGHT - 50):
                    use_fingertip = False

                # Final hand position (fingertip or palm)
                if use_fingertip:
                    hand_x = fingertip_x
                    hand_y = fingertip_y
                else:
                    hand_x = palm_x
                    hand_y = palm_y

                # Assign to left/right hand
                if handedness == 'Left':
                    left_hand_x = hand_x
                elif handedness == 'Right':
                    right_hand_x = hand_x

                if main_hand_y is None:
                    main_hand_y = hand_y

        # Determine basket position
        if left_hand_x is not None and right_hand_x is not None:
            current_basket_x = (left_hand_x + right_hand_x) // 2
            current_basket_y = main_hand_y
        elif left_hand_x is not None:
            current_basket_x = left_hand_x
            current_basket_y = main_hand_y
        elif right_hand_x is not None:
            current_basket_x = right_hand_x
            current_basket_y = main_hand_y
        else:
            target_center_x = SCREEN_WIDTH // 2
            target_center_y = SCREEN_HEIGHT - 100
            current_basket_x = int(prev_x * 0.9 + target_center_x * 0.1)
            current_basket_y = int(prev_y * 0.9 + target_center_y * 0.1)

        # Fixed basket width (but can be made dynamic)
        current_basket_width = BASKET_INITIAL_WIDTH

        # Apply Y-offset to position basket below hand
        basket_y_offset = BASKET_HEIGHT // 4
        current_basket_y += basket_y_offset

        prev_x = current_basket_x
        prev_y = current_basket_y
        prev_width = current_basket_width

        basket.update_position(current_basket_x, current_basket_y, current_basket_width)

        # Draw stars (unchanged)
        for i, (x, y, size, brightness) in enumerate(stars):
            brightness = max(0.3, min(1.0, brightness + random.uniform(-0.1, 0.1)))
            stars[i] = (x, y, size, brightness)
            cv2.circle(frame, (x, y), size, (int(255*brightness),)*3, -1)

        # Update and draw balls (unchanged)
        for ball in balls[:]:
            ball.update()
            ball.draw(frame)
            catch_zone_x_min = basket.x - basket.width // 4
            catch_zone_x_max = basket.x + basket.width // 4
            catch_zone_y_min = basket.y - basket.height // 2
            catch_zone_y_max = basket.y - basket.height // 2 + 30
            if (catch_zone_x_min < ball.x < catch_zone_x_max and
                catch_zone_y_min < ball.y < catch_zone_y_max):
                balls.remove(ball)
                SCORE += ball.points
                try:
                    catch_sound.play()
                except:
                    pass
                basket.add_particles(25)
            if ball.y > SCREEN_HEIGHT + ball.size:
                balls.remove(ball)

        basket.draw(frame)

        # Display game info (unchanged)
        cv2.putText(frame, f"Score: {SCORE}", (10, 30),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Time: {int(remaining_time)}s", (SCREEN_WIDTH - 150, 30),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Level: {level}", (SCREEN_WIDTH//2 - 50, 30),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)

        # Game over handling (unchanged)
        if remaining_time <= 0:
            if not game_over:
                if not sound_played:
                    try:
                        game_over_sound.play()
                        background_music.stop()
                    except:
                        pass
                    sound_played = True
                game_over = True

            overlay = frame.copy()
            cv2.circle(overlay, (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), SCREEN_HEIGHT//2, (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            for _ in range(5):
                x = random.randint(0, SCREEN_WIDTH)
                y = random.randint(0, SCREEN_HEIGHT)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.circle(frame, (x, y), 5, color, -1)

            trophy.update()
            trophy.draw(frame)

            congrats_size = cv2.getTextSize("CONGRATULATIONS!", cv2.FONT_HERSHEY_COMPLEX, 1.5, 3)[0]
            score_size = cv2.getTextSize(f"Score: {SCORE}", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
            won_size = cv2.getTextSize("You Won!", cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 2)[0]
            play_again_size = cv2.getTextSize("Press SPACE to play again", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

            text_x_congrats = (SCREEN_WIDTH - congrats_size[0]) // 2
            text_x_score = (SCREEN_WIDTH - score_size[0]) // 2
            text_x_won = (SCREEN_WIDTH - won_size[0]) // 2
            text_x_play_again = (SCREEN_WIDTH - play_again_size[0]) // 2

            text_y = SCREEN_HEIGHT // 2 - 100

            cv2.putText(frame, "CONGRATULATIONS!", (text_x_congrats, text_y),
               cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Score: {SCORE}", (text_x_score, text_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "You Won!", (text_x_won, text_y + 120),
               cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press SPACE to play again", (text_x_play_again, text_y + 180),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("--- DEBUG: FAILED TO ENCODE FRAME TO JPEG! ---")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/score')
def get_score():
    global SCORE, game_over
    return jsonify({'score': SCORE, 'game_over': game_over})

@app.route('/restart', methods=['POST'])
def restart_game():
    global balls, SCORE, start_time, game_over, sound_played, level, prev_x, prev_y, prev_width, trophy
    balls = []
    SCORE = 0
    start_time = time.time()
    game_over = False
    sound_played = False
    level = 1
    trophy = Trophy()
    prev_x = SCREEN_WIDTH // 2
    prev_y = SCREEN_HEIGHT - 100
    prev_width = BASKET_INITIAL_WIDTH
    try:
        background_music.play(-1)
    except:
        pass
    return jsonify({'status': 'success'})

if __name__ == "__main__":
    print("DEBUG: Starting Flask app...")
    app.run(debug=True)