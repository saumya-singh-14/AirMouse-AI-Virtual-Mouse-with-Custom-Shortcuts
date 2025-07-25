import cv2
import numpy as np
import time
import pyautogui
import sqlite3
import webbrowser
import os
import subprocess
import mediapipe as mp
import HandTrackingModule as htm
import urllib.parse # Needed for URL encoding landmarks

# Configuration constants
FLAG_FILE = "resume_vmouse.flag" # Define flag file name (must match api.py)
TEMP_IMAGE_FILE = "temp_gesture.jpg" # Temporary file for captured image
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 10  # smoothening for better control
scroll_sensitivity = 15
COOLDOWN_TIME = 5  # cooldown to avoid frequent triggers
MATCH_THRESHOLD = 0.5  # Threshold for gesture matching accuracy
MIN_GESTURE_DURATION = 1.0  # Increased time to recognize a gesture

# Initialize mediapipe for gesture shortcuts
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variables for mouse control
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
last_triggered = {}  # Stores last execution time for each gesture
gesture_start_time = 0
current_gesture = None

def load_shortcuts():
    """Load gesture shortcuts from the database"""
    conn = sqlite3.connect("shortcuts.db")
    cursor = conn.cursor()
    cursor.execute("SELECT gesture, command, landmarks FROM shortcuts")
    data = {row[0]: (row[1], eval(row[2])) for row in cursor.fetchall()}
    conn.close()
    return data

def execute_command(command):
    """Execute the command associated with a recognized gesture"""
    print(f"Executing: {command}")
    time.sleep(1)  # Small delay before executing to prevent accidental triggers
    if command.startswith("http"):
        webbrowser.open(command)
    elif os.path.isdir(command) or os.path.isfile(command):
        os.startfile(command)
    else:
        try:
            subprocess.Popen(command, shell=True)
        except Exception as e:
            print(f"Error executing command: {e}")

def run_api():
    """Run the api.py file as a separate process, open the frontend, and return the process."""
    process = None
    try:
        # Start the Flask app
        print("Starting api.py process...")
        # Use shell=True on Windows if 'python' isn't directly in PATH for subprocess
        # Or ensure your python env is correctly set up in PATH
        process = subprocess.Popen(['python', 'api.py'], shell=True)
        print(f"api.py started with PID: {process.pid}")
        time.sleep(2) # Give server time to start
    except Exception as e:
        print(f"Error starting api.py process: {e}")
    return process # Return the process object

def open_capture_details_browser(image_path, landmarks_str):
    """Opens the browser to the capture details page."""
    try:
        # URL encode the landmarks string to handle special characters safely
        encoded_landmarks = urllib.parse.quote(landmarks_str)
        # Pass image path relative to where api.py serves images from
        image_filename = os.path.basename(image_path) 
        url = f'http://127.0.0.1:5000/capture_details?image_file={image_filename}&landmarks={encoded_landmarks}'
        print(f"Opening browser to: {url}")
        webbrowser.open(url)
        return True
    except Exception as e:
        print(f"Error opening browser: {e}")
        return False

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist position"""
    base = np.array(landmarks[0])  # Use wrist as the base
    return [(x - base[0], y - base[1], z - base[2]) for x, y, z in landmarks]

def is_match(live_landmarks, saved_landmarks):
    """Check if live landmarks match saved gesture landmarks"""
    if len(live_landmarks) != len(saved_landmarks):
        return False
    
    live_landmarks = normalize_landmarks(live_landmarks)
    saved_landmarks = normalize_landmarks(saved_landmarks)
    
    distance = np.linalg.norm(np.array(live_landmarks) - np.array(saved_landmarks))
    confidence = 1 - (distance / MATCH_THRESHOLD)  # Compute confidence score

    print(f"Gesture confidence: {confidence:.2f}")
    
    return confidence > 0.5  # Only match if confidence is above 50%

def main():
    global pTime, plocX, plocY, clocX, clocY, gesture_start_time, current_gesture
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    
    # Initialize hand detectors
    detector = htm.handDetector(maxHands=1)
    wScr, hScr = pyautogui.size()
    shortcuts = load_shortcuts()
    print("Loaded Shortcuts:", shortcuts)
    
    # Reduce mouse movement speed
    pyautogui.PAUSE = 0.01
    pyautogui.MINIMUM_SLEEP = 0.01
    pyautogui.MINIMUM_DURATION = 0.01
    
    with mp_hands.Hands(
        min_detection_confidence=0.8, 
        min_tracking_confidence=0.8,
        max_num_hands=1  # Only detect one hand for gestures
    ) as gesture_hands:
        print("\nShow a gesture to execute a shortcut or use index finger for mouse control!")
        
        while True:
            # Read frame once
            success, img = cap.read()
            if not success:
                continue
                
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process for gesture recognition first
            gesture_detected = False
            gesture_result = gesture_hands.process(img_rgb)
            
            if gesture_result.multi_hand_landmarks:
                for hand_landmarks in gesture_result.multi_hand_landmarks:
                    live_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    for gesture, (command, saved_landmarks) in shortcuts.items():
                        current_time = time.time()
                        if gesture in last_triggered and (current_time - last_triggered[gesture] < COOLDOWN_TIME):
                            continue
                        
                        if is_match(live_landmarks, saved_landmarks):
                            last_triggered[gesture] = current_time
                            print(f"Matched Gesture: {gesture}")
                            execute_command(command)
                            gesture_detected = True
                            break            
            # Only proceed with mouse control if no gesture was detected
            if not gesture_detected:
                success, img = cap.read()
                img = detector.findHands(img)
                lmList, bbox = detector.findPosition(img)

                # Step2: Get the tip of the index and middle finger
                if len(lmList) != 0:
                    x1, y1 = lmList[8][1:]    # index finger
                    x2, y2 = lmList[12][1:]   #middle finger
                    x3, y3 = lmList[4][1:]   # Thumb

                    # Step3: Check which fingers are up
                    fingers = detector.fingersUp()
                    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                                (255, 0, 255), 2)

                    # Step4: Only Index Finger: Moving Mode
                    if fingers[1] == 1 and fingers[2] == 0:

                        # Step5: Convert the coordinates
                        x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

                        # Step6: Smooth Values
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening

                        # Step7: Move Mouse
                        pyautogui.moveTo(wScr - clocX, clocY)
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    # Step8: Both Index and middle are up: Clicking Mode
                    if fingers[1] == 1 and fingers[2] == 1:

                        # Step9: Find distance between fingers
                        length, img, lineInfo = detector.findDistance(8, 12, img)

                        # Step10: Click mouse if distance short
                        if length < 40:
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            pyautogui.click()
                    
                    # Step9: Right Click - Pinch Gesture (Index & Thumb Close)
                    if fingers[1] == 1 and fingers[0] == 1:
                        length, img, lineInfo = detector.findDistance(8, 4, img)  # Index & Thumb finger
                        if length < 40:  # If fingers are pinched
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            pyautogui.click(button='right')

                    # Step10: Scrolling - Index & Middle Fingers Up
                    if fingers[1] == 1 and fingers[2] == 1:
                        diffY = y2 - y1  # Difference in Y-axis position of Index & Middle
                        if diffY > 20:  # Move Down
                            pyautogui.scroll(-scroll_sensitivity)
                        elif diffY < -20:  # Move Up
                            pyautogui.scroll(scroll_sensitivity)

            # Frame rate calculation
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, f"FPS: {int(fps)}", (28, 58), cv2.FONT_HERSHEY_PLAIN, 2, (255, 8, 8), 2)
            
            # Display mode information
            mode_text = "Gesture Mode" if gesture_detected else "Mouse Control"
            cv2.putText(img, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add button to run api.py
            cv2.putText(img, "Press 'r' to run api.py", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check for key press *before* processing/drawing potentially blocking imshow
            key = cv2.waitKey(1) & 0xFF

            # --- 's' key: Capture Gesture and Launch Details UI ---
            if key == ord('s'):
                print("Save key pressed.")
                captured_landmarks = None
                # Check if hands are detected in the current frame
                if gesture_result.multi_hand_landmarks:
                    # Use the first detected hand's landmarks
                    hand_landmarks_for_save = gesture_result.multi_hand_landmarks[0]
                    captured_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks_for_save.landmark]
                    print("Hand landmarks captured.")
                
                if captured_landmarks:
                    # Save the current frame (img) temporarily
                    temp_image_path = os.path.join(os.getcwd(), TEMP_IMAGE_FILE) # Save in CWD
                    save_success = cv2.imwrite(temp_image_path, img)

                    if save_success:
                        print(f"Temporary gesture image saved to {temp_image_path}")
                        
                        # Release camera and close window BEFORE starting API/Browser
                        print("Releasing camera...")
                        if cap.isOpened():
                            cap.release()
                        cv2.destroyAllWindows() 
                        
                        # Convert landmarks to string for URL transfer
                        landmarks_str = str(captured_landmarks)

                        # Ensure flag file doesn't exist from a previous run
                        if os.path.exists(FLAG_FILE):
                            try:
                                os.remove(FLAG_FILE)
                            except Exception as e:
                                print(f"Warning: Could not remove pre-existing flag file: {e}")

                        # Start API process
                        api_process = run_api() 

                        if api_process:
                            # Open browser to capture details page
                            browser_opened = open_capture_details_browser(temp_image_path, landmarks_str)

                            if browser_opened:
                                print("Waiting for gesture save signal from web UI...")
                                # --- Start Waiting Loop ---
                                while True:
                                    if os.path.exists(FLAG_FILE):
                                        print("Resume signal received.")
                                        try:
                                            os.remove(FLAG_FILE)
                                            print("Flag file removed.")
                                        except Exception as e:
                                            print(f"Error removing flag file: {e}")
                                        
                                        # Terminate the Flask server process
                                        try:
                                            print(f"Terminating api.py process (PID: {api_process.pid})...")
                                            api_process.terminate()
                                            api_process.wait(timeout=5) 
                                            if api_process.poll() is None:
                                                 print("Force killing api.py process...")
                                                 api_process.kill()
                                        except Exception as e:
                                            print(f"Error terminating api.py process: {e}")

                                        # Remove temp image file
                                        try:
                                            if os.path.exists(temp_image_path):
                                                os.remove(temp_image_path)
                                                print(f"Temporary image {temp_image_path} removed.")
                                        except Exception as e:
                                            print(f"Warning: Could not remove temporary image file: {e}")

                                        # Re-initialize camera and necessary components
                                        print("Re-initializing camera...")
                                        cap = cv2.VideoCapture(0)
                                        if not cap.isOpened():
                                            print("Error: Failed to re-open camera. Exiting.")
                                            break # Break inner wait loop -> leads to outer break
                                        cap.set(3, wCam)
                                        cap.set(4, hCam)
                                        pTime = time.time() 
                                        plocX, plocY = 0, 0 
                                        clocX, clocY = 0, 0
                                        print("Resuming virtual mouse...")
                                        break # Break inner wait loop and continue outer loop
                                    
                                    # Check if the API process has terminated unexpectedly
                                    if api_process.poll() is not None:
                                        print("API process terminated unexpectedly. Exiting wait.")
                                        # Attempt to remove temp image file even if API died
                                        try:
                                            if os.path.exists(temp_image_path):
                                                os.remove(temp_image_path)
                                                print(f"Temporary image {temp_image_path} removed.")
                                        except Exception as e:
                                            print(f"Warning: Could not remove temporary image file: {e}")
                                        break # Break inner wait loop -> leads to outer break

                                    time.sleep(1) # Check every second
                                # --- End Waiting Loop ---

                                if not cap.isOpened(): # If camera failed to re-open after wait
                                     break # Break outer main loop
                                
                                continue # Continue the main loop to start processing frames again

                            else: # Browser failed to open
                                print("Failed to open browser. Terminating API process.")
                                if api_process.poll() is None:
                                    api_process.terminate()
                                # Attempt to remove temp image file
                                try:
                                    if os.path.exists(temp_image_path):
                                        os.remove(temp_image_path)
                                except Exception as e:
                                    print(f"Warning: Could not remove temporary image file: {e}")
                                # Re-initialize camera and continue
                                print("Attempting to resume virtual mouse...")
                                cap = cv2.VideoCapture(0)
                                if cap.isOpened():
                                     cap.set(3, wCam)
                                     cap.set(4, hCam)
                                     pTime = time.time()
                                else:
                                     print("Error: Failed to re-open camera after browser failure. Exiting.")
                                     break # Exit main loop

                        else: # API process failed to start
                            print("Failed to start API process.")
                            # Attempt to remove temp image file
                            try:
                                if os.path.exists(temp_image_path):
                                    os.remove(temp_image_path)
                            except Exception as e:
                                print(f"Warning: Could not remove temporary image file: {e}")
                            # No need to re-initialize camera as it wasn't released
                            print("Continuing virtual mouse...")

                    else: # Failed to save temp image
                        print("Error: Failed to save temporary gesture image.")
                        # Continue without launching API/Browser

                else: # No hand detected
                    print("No hand detected. Cannot save gesture.")

            # --- 'r' key: Launch API index page (non-blocking) ---
            elif key == ord('r'):
                print("Launch API index key pressed.")
                # Start API process (doesn't check if already running)
                run_api() 
                # Open browser to index page
                try:
                    print("Opening browser to API index...")
                    webbrowser.open('http://127.0.0.1:5000/')
                except Exception as e:
                    print(f"Error opening browser for index: {e}")
                # NOTE: This does not stop merged_vmouse.py

            # --- 'q' key: Quit ---
            elif key == ord('q'):
                print("Quit key pressed. Exiting.")
                break # Exit the main loop

            # --- Display ---
            # Only display if the camera is presumably open
            if cap.isOpened():
                 cv2.imshow("AI Virtual Mouse with Gesture Shortcuts", img)
            else:
                 print("Camera not open, cannot display frame.")
                 # Maybe wait a bit and try to reopen? Or just exit? Let's exit for now.
                 break


    # --- Cleanup ---
    print("Cleaning up resources...")
    # Ensure camera is released
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Camera released.")
    cv2.destroyAllWindows()
    print("OpenCV windows destroyed.")

    # Ensure flag file is removed on exit (if it exists)
    if os.path.exists(FLAG_FILE):
        try:
            os.remove(FLAG_FILE)
            print("Flag file removed on exit.")
        except Exception as e:
            print(f"Warning: Could not remove flag file on exit: {e}")

    # Ensure temp image file is removed on exit (if it exists)
    temp_image_path_cleanup = os.path.join(os.getcwd(), TEMP_IMAGE_FILE)
    if os.path.exists(temp_image_path_cleanup):
        try:
            os.remove(temp_image_path_cleanup)
            print(f"Temporary image {temp_image_path_cleanup} removed on exit.")
        except Exception as e:
            print(f"Warning: Could not remove temporary image file on exit: {e}")

    # Ensure api_process is terminated if it was started and might still be running
    # This handles cases where 'q' is pressed while waiting for the flag
    if 'api_process' in locals() and api_process is not None and api_process.poll() is None:
         print(f"Terminating leftover api.py process (PID: {api_process.pid})...")
         try:
             api_process.terminate()
             api_process.wait(timeout=2)
             if api_process.poll() is None:
                 api_process.kill()
             print("Leftover api.py process terminated.")
         except Exception as e:
             print(f"Error terminating leftover api.py process: {e}")

if __name__ == "__main__":
    main()
