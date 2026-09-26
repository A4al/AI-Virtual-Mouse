import cv2
import mediapipe as mp
import pyautogui
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pycaw.pycaw import AudioUtilities


# =========================================================
# 1. SCREEN SIZE
# =========================================================

screen_width, screen_height = pyautogui.size()

print("Screen:", screen_width, "x", screen_height)


# =========================================================
# 2. LOAD MEDIAPIPE MODEL
# =========================================================

base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# =========================================================
# 3. WINDOWS AUDIO DEVICE
# =========================================================

device = AudioUtilities.GetSpeakers()

print("Audio device:", device.FriendlyName)
print("Current volume:", device.volume_percent)


# =========================================================
# 4. OPEN CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened!")

    exit()


# =========================================================
# 5. VARIABLES
# =========================================================

# ---------------------------------------------------------
# Click tracking
# ---------------------------------------------------------

last_left_click = 0
last_right_click = 0

left_gesture_frames = 0
right_gesture_frames = 0

required_frames = 2

click_cooldown = 0.5


# ---------------------------------------------------------
# Scroll tracking
# ---------------------------------------------------------

previous_scroll_y = None


# ---------------------------------------------------------
# Volume tracking
# ---------------------------------------------------------

previous_volume_y = None


# ---------------------------------------------------------
# Cursor smoothing
# ---------------------------------------------------------

previous_x = 0
previous_y = 0

smoothing = 0.3


# ---------------------------------------------------------
# Drag tracking
# ---------------------------------------------------------

dragging = False

drag_start_time = 0

drag_hold_time = 0.4


# =========================================================
# 6. MAIN LOOP
# =========================================================

while True:
    mode = "NO HAND"

    ret, frame = cap.read()

    if not ret:

        print("Could not read camera frame.")

        break


    # =====================================================
    # MIRROR CAMERA
    # =====================================================

    frame = cv2.flip(frame, 1)


    # =====================================================
    # CONVERT BGR → RGB
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # MEDIAPIPE IMAGE
    # =====================================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # =====================================================
    # DETECT HAND
    # =====================================================

    result = detector.detect(mp_image)


    # =====================================================
    # HAND DETECTED
    # =====================================================
    mode = "NO HAND"
    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # =================================================
        # 7. LANDMARKS
        # =================================================

        thumb = hand[4]

        index_finger = hand[8]

        middle_finger = hand[12]

        ring_finger = hand[16]


        # =================================================
        # 8. FINGER EXTENSION
        # =================================================

        index_extended = (
            index_finger.y < hand[6].y
        )

        middle_extended = (
            middle_finger.y < hand[10].y
        )


        # =================================================
        # 9. CALCULATE GESTURE DISTANCES
        # =================================================

        # -------------------------------------------------
        # Thumb + Index
        # -------------------------------------------------

        left_distance = math.sqrt(
            (thumb.x - index_finger.x) ** 2 +
            (thumb.y - index_finger.y) ** 2
        )


        # -------------------------------------------------
        # Thumb + Middle
        # -------------------------------------------------

        right_distance = math.sqrt(
            (thumb.x - middle_finger.x) ** 2 +
            (thumb.y - middle_finger.y) ** 2
        )


        # -------------------------------------------------
        # Thumb + Ring
        # -------------------------------------------------

        volume_distance = math.sqrt(
            (thumb.x - ring_finger.x) ** 2 +
            (thumb.y - ring_finger.y) ** 2
        )


        # =================================================
        # 10. DETECT GESTURES
        # =================================================

        volume_gesture = (
            volume_distance < 0.08
        )


        scroll_gesture = (
            index_extended and
            middle_extended
        )


        right_click_gesture = (
            right_distance < 0.03
        )


        left_click_gesture = (
            left_distance < 0.05
        )


        # =================================================
        # 11. INDEX POSITION
        # =================================================

        x = index_finger.x

        y = index_finger.y


        # =================================================
        # 12. GESTURE PRIORITY
        # =================================================
        #
        # Priority:
        #
        # 1. Volume
        # 2. Scroll
        # 3. Right Click
        # 4. Left Click / Drag
        # 5. Cursor
        #
        # =================================================


        # =================================================
        # MODE 1 → VOLUME
        # =================================================

        if volume_gesture:
            mode = "VOLUME"

            # Reset other gesture tracking

            previous_scroll_y = None

            previous_volume_y = (
               previous_volume_y
                if previous_volume_y is not None
                else ring_finger.y
            )

            left_gesture_frames = 0

            right_gesture_frames = 0

            # ---------------------------------------------
            # Volume movement
            # ---------------------------------------------

            volume_movement = (
                ring_finger.y -
                previous_volume_y
            )


            # ---------------------------------------------
            # MOVE UP → VOLUME UP
            # ---------------------------------------------

            if volume_movement < -0.015:

                current_volume = (
                    device.volume_percent / 100
                )

                new_volume = min(
                    current_volume + 0.05,
                    1.0
                )

                device.EndpointVolume.SetMasterVolumeLevelScalar(
                    new_volume,
                    None
                )

                print(
                    "VOLUME UP:",
                    round(new_volume * 100),
                    "%"
                )

                previous_volume_y = ring_finger.y


            # ---------------------------------------------
            # MOVE DOWN → VOLUME DOWN
            # ---------------------------------------------

            elif volume_movement > 0.015:

                current_volume = (
                    device.volume_percent / 100
                )

                new_volume = max(
                    current_volume - 0.05,
                    0.0
                )

                device.EndpointVolume.SetMasterVolumeLevelScalar(
                    new_volume,
                    None
                )

                print(
                    "VOLUME DOWN:",
                    round(new_volume * 100),
                    "%"
                )

                previous_volume_y = ring_finger.y


            # ---------------------------------------------
            # Display mode
            # ---------------------------------------------

            cv2.putText(
                frame,
                "VOLUME MODE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )


        # =================================================
        # MODE 2 → SCROLL
        # =================================================

        elif scroll_gesture:
            mode = "SCROLL"

            # Reset other gesture tracking

            previous_volume_y = None

            left_gesture_frames = 0

            right_gesture_frames = 0


            # ---------------------------------------------
            # Start scroll tracking
            # ---------------------------------------------

            if previous_scroll_y is None:

                previous_scroll_y = y

            else:

                movement = (
                    y - previous_scroll_y
                )


                # -----------------------------------------
                # Hand DOWN → Scroll DOWN
                # -----------------------------------------

                if movement > 0.04:

                    pyautogui.scroll(-8)

                    previous_scroll_y = y

                    print("SCROLL DOWN")


                # -----------------------------------------
                # Hand UP → Scroll UP
                # -----------------------------------------

                elif movement < -0.04:

                    pyautogui.scroll(8)

                    previous_scroll_y = y

                    print("SCROLL UP")


            # ---------------------------------------------
            # Display mode
            # ---------------------------------------------

            cv2.putText(
                frame,
                "SCROLL MODE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        # =================================================
        # MODE 3 → RIGHT CLICK
        # =================================================

        elif right_click_gesture:
            mode = "RIGHT CLICK"

            # Reset other gesture tracking

            previous_scroll_y = None

            previous_volume_y = None

            left_gesture_frames = 0


            # ---------------------------------------------
            # Count stable frames
            # ---------------------------------------------

            right_gesture_frames += 1


            # ---------------------------------------------
            # Stable gesture reached
            # ---------------------------------------------

            if right_gesture_frames >= required_frames:

                current_time = time.time()


                # -----------------------------------------
                # Cooldown check
                # -----------------------------------------

                if (
                    current_time - last_right_click
                    > click_cooldown
                ):

                    pyautogui.rightClick()

                    last_right_click = current_time

                    print("RIGHT CLICK")

                    right_gesture_frames = 0


            # ---------------------------------------------
            # Display mode
            # ---------------------------------------------

            cv2.putText(
                frame,
                "RIGHT CLICK",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )


        # =================================================
        # MODE 4 → LEFT CLICK / DRAG
        # =================================================

        elif left_click_gesture:

            # Reset other gesture tracking

            previous_scroll_y = None

            previous_volume_y = None

            right_gesture_frames = 0


            # ---------------------------------------------
            # Start pinch timer
            # ---------------------------------------------

            if left_gesture_frames == 0:

                drag_start_time = time.time()


            # Count consecutive pinch frames

            left_gesture_frames += 1


            current_time = time.time()


            pinch_duration = (
                current_time -
                drag_start_time
            )


            # =================================================
            # HOLD PINCH → DRAG
            # =================================================

            if pinch_duration >= drag_hold_time:
                mode = "DRAGGING"

                # -----------------------------------------
                # Start dragging
                # -----------------------------------------

                if not dragging:

                    pyautogui.mouseDown()

                    dragging = True

                    print("DRAG START")


                # -----------------------------------------
                # Move cursor while dragging
                # -----------------------------------------

                target_x = int(
                    x * screen_width
                )

                target_y = int(
                    y * screen_height
                )


                smooth_x = (
                    previous_x +
                    (target_x - previous_x)
                    * smoothing
                )

                smooth_y = (
                    previous_y +
                    (target_y - previous_y)
                    * smoothing
                )


                pyautogui.moveTo(
                    int(smooth_x),
                    int(smooth_y)
                )


                previous_x = smooth_x

                previous_y = smooth_y


                # -----------------------------------------
                # Display drag mode
                # -----------------------------------------

                cv2.putText(
                    frame,
                    "DRAGGING",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )


            # =================================================
            # QUICK PINCH → LEFT CLICK
            # =================================================

            elif (
                left_gesture_frames >= required_frames
                and pinch_duration < drag_hold_time
            ):

                # We DON'T click here.
                #
                # We wait until the pinch is released.
                #
                # This prevents a quick pinch from becoming
                # both a click and a drag.


                cv2.putText(
                    frame,
                    "PINCH",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )


        # =================================================
        # MODE 5 → NORMAL CURSOR
        # =================================================

        else:
            mode = "CURSOR"

            # ---------------------------------------------
            # If we were dragging, release mouse
            # ---------------------------------------------

            if dragging:

                pyautogui.mouseUp()

                dragging = False

                print("DRAG END")


            # ---------------------------------------------
            # Quick pinch release → LEFT CLICK
            # ---------------------------------------------

            elif (
                left_gesture_frames >= required_frames
            ):

                current_time = time.time()


                if (
                    current_time - last_left_click
                    > click_cooldown
                ):

                    pyautogui.click()

                    last_left_click = current_time

                    print("LEFT CLICK")


            # ---------------------------------------------
            # Reset left-click / drag tracking
            # ---------------------------------------------

            left_gesture_frames = 0

            drag_start_time = 0


            # ---------------------------------------------
            # Reset scroll / volume tracking
            # ---------------------------------------------

            previous_scroll_y = None

            previous_volume_y = None


            # ---------------------------------------------
            # Normal cursor movement
            # ---------------------------------------------

            target_x = int(
                x * screen_width
            )

            target_y = int(
                y * screen_height
            )


            smooth_x = (
                previous_x +
                (target_x - previous_x)
                * smoothing
            )

            smooth_y = (
                previous_y +
                (target_y - previous_y)
                * smoothing
            )


            pyautogui.moveTo(
                int(smooth_x),
                int(smooth_y)
            )


            previous_x = smooth_x

            previous_y = smooth_y
            # =================================================
        # 13. PROFESSIONAL HAND SKELETON
        # =================================================

        h, w, _ = frame.shape
        cv2.putText(
    frame,
    "GESTURES",
    (w - 235, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)


        # -------------------------------------------------
        # MediaPipe hand connections
        # -------------------------------------------------

        hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),

            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),

            # Middle finger
            (5, 9), (9, 10), (10, 11), (11, 12),

            # Ring finger
            (9, 13), (13, 14), (14, 15), (15, 16),

            # Pinky finger
            (13, 17), (17, 18), (18, 19), (19, 20),

            # Palm
            (0, 17)
        ]


        # -------------------------------------------------
        # Convert all 21 landmarks to pixel positions
        # -------------------------------------------------

        landmark_points = []

        for landmark in hand:

            px = int(landmark.x * w)
            py = int(landmark.y * h)

            landmark_points.append((px, py))


        # -------------------------------------------------
        # Draw green skeleton lines
        # -------------------------------------------------

        for start_point, end_point in hand_connections:

            cv2.line(
                frame,
                landmark_points[start_point],
                landmark_points[end_point],
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )


        # -------------------------------------------------
        # Draw small professional joints
        # -------------------------------------------------

        for point in landmark_points:

            # Dark outline
            cv2.circle(
                frame,
                point,
                5,
                (0, 80, 0),
                -1,
                cv2.LINE_AA
            )

            # Bright green center
            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 0),
                -1,
                cv2.LINE_AA
            )


        # -------------------------------------------------
        # Highlight index fingertip
        # -------------------------------------------------

        cv2.circle(
            frame,
            landmark_points[8],
            8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


       
    # =====================================================
    # NO HAND DETECTED
    # =====================================================

    else:

        # If hand disappears while dragging,
        # release the mouse button safely.

        if dragging:

            pyautogui.mouseUp()

            dragging = False

            print("DRAG END")


        previous_scroll_y = None

        previous_volume_y = None

        left_gesture_frames = 0

        right_gesture_frames = 0

        drag_start_time = 0

            # =================================================
        # HUD → VIRTUAL MOUSE TITLE
        # =================================================

        cv2.rectangle(
            frame,
            (20, 15),
            (260, 65),
            (20, 20, 20),
            -1
        )

        cv2.rectangle(
            frame,
            (20, 15),
            (260, 65),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "VIRTUAL MOUSE",
            (35, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        
        cv2.putText(
    frame,
    "MODE: " + mode,
    (35, 95),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2,
    cv2.LINE_AA
) 
    
    instruction = {
       "CURSOR": "Move index finger",
       "LEFT CLICK": "Pinch thumb + index",
       "RIGHT CLICK": "Thumb + middle",
       "SCROLL": "Move index + middle",
       "VOLUME": "Thumb + ring",
       "DRAGGING": "Hold pinch",
       "NO HAND": "Show your hand"
    }

    cv2.putText(
    frame,
    instruction.get(mode, ""),
    (35, 125),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    1,
    cv2.LINE_AA
)
    # =====================================================
    # 18. SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "Virtual Mouse",
        frame
    )


    # =====================================================
    # 19. PRESS Q TO EXIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================================
# 20. CLEANUP
# =========================================================

# Safety: release mouse if program exits during dragging

if dragging:

    pyautogui.mouseUp()


cap.release()

cv2.destroyAllWindows()

detector.close()