from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import time
import sqlite3
import mediapipe as mp
import numpy as np
import ast # For safely evaluating landmarks string
import shutil # For moving the temp image file
import urllib.parse # For decoding landmarks

app = Flask(__name__)
UPLOAD_FOLDER = "gesture_images"
TEMP_IMAGE_FILE = "temp_gesture.jpg" # Must match merged_vmouse.py
FLAG_FILE = "resume_vmouse.flag" # Define flag file name
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

@app.route("/")
def index():
    # Ensure flag file is removed if we visit the index page,
    # in case a previous save attempt was abandoned.
    if os.path.exists(FLAG_FILE):
        try:
            os.remove(FLAG_FILE)
            print(f"Removed stale flag file '{FLAG_FILE}' on index load.")
        except Exception as e:
            print(f"Warning: Could not remove stale flag file: {e}")
    return render_template("index.html")


@app.route("/capture_details")
def capture_details():
    """Displays the captured image and form to enter gesture details."""
    image_file = request.args.get('image_file')
    landmarks_encoded = request.args.get('landmarks')

    if not image_file or not landmarks_encoded:
        return "Error: Missing image file or landmarks.", 400

    # Decode landmarks (handle potential errors)
    try:
        landmarks_str = urllib.parse.unquote(landmarks_encoded)
    except Exception as e:
        print(f"Error decoding landmarks: {e}")
        return "Error: Could not decode landmarks.", 400

    # Check if the temporary image file exists
    temp_image_path = os.path.join(os.getcwd(), image_file) # Assuming it's in CWD
    if not os.path.exists(temp_image_path):
         # Maybe it's already been moved? Or never created?
         # Let's check the final destination too just in case of refresh
         final_image_path = os.path.join(UPLOAD_FOLDER, image_file)
         if not os.path.exists(final_image_path):
              print(f"Error: Temporary image file not found: {temp_image_path}")
              return f"Error: Image file '{image_file}' not found.", 404
         else: # Serve from final destination if already moved
              image_file_to_render = image_file # Render the one in UPLOAD_FOLDER
              print(f"Warning: Temp image not found, using existing final image: {final_image_path}")
    else:
         image_file_to_render = image_file # Render the one in CWD (temp)

    return render_template("capture_details.html", 
                           image_file=image_file_to_render, 
                           landmarks_str=landmarks_str,
                           temp_image_filename=image_file) # Pass original temp filename


@app.route("/finalize_save", methods=["POST"])
def finalize_save():
    """Saves the gesture details from the form to the database."""
    gesture_name = request.form.get('gesture_name')
    command = request.form.get('command')
    landmarks_str = request.form.get('landmarks_str')
    temp_image_filename = request.form.get('temp_image_filename')

    if not all([gesture_name, command, landmarks_str, temp_image_filename]):
        return "Error: Missing form data.", 400

    # Define paths
    temp_image_path = os.path.join(os.getcwd(), temp_image_filename)
    # Sanitize gesture_name for filename? For now, use as is.
    final_image_filename = f"{gesture_name}.jpg"
    final_image_path = os.path.join(UPLOAD_FOLDER, final_image_filename)

    try:
        # 1. Move the temporary image to the final destination
        if os.path.exists(temp_image_path):
            shutil.move(temp_image_path, final_image_path)
            print(f"Moved {temp_image_path} to {final_image_path}")
        elif not os.path.exists(final_image_path):
             # If temp doesn't exist AND final doesn't exist, something is wrong
             print(f"Error: Neither temp ({temp_image_path}) nor final ({final_image_path}) image found.")
             return "Error: Image file lost.", 500
        else:
             print(f"Warning: Temp image {temp_image_path} not found, assuming already moved to {final_image_path}.")


        # 2. Parse landmarks string safely
        # The string looks like '[(x,y,z), (x,y,z), ...]'
        landmarks = ast.literal_eval(landmarks_str)
        if not isinstance(landmarks, list): # Basic validation
             raise ValueError("Parsed landmarks are not a list.")
        print("Landmarks parsed successfully.")

        # 3. Save to database
        conn = sqlite3.connect("shortcuts.db")
        cursor = conn.cursor()
        # Use final_image_path (relative path within project) for DB
        db_image_path = os.path.join(UPLOAD_FOLDER, final_image_filename).replace("\\", "/") # Use forward slashes for consistency
        cursor.execute(
            "INSERT OR REPLACE INTO shortcuts (gesture, command, landmarks, image) VALUES (?, ?, ?, ?)",
            (gesture_name, command, str(landmarks), db_image_path) # Store the final path
        )
        conn.commit()
        conn.close()
        print(f"Gesture '{gesture_name}' saved to database.")

        # 4. Create the flag file to signal merged_vmouse.py
        with open(FLAG_FILE, 'w') as f:
            pass # Create an empty file
        print(f"Flag file '{FLAG_FILE}' created.")

        # 5. Render a success page
        return render_template("save_success.html", gesture_name=gesture_name)

    except sqlite3.Error as db_err:
        print(f"Database error: {db_err}")
        return f"Database error: {db_err}", 500
    except FileNotFoundError as fnf_err:
         print(f"File not found error during move: {fnf_err}")
         return f"File not found error: {fnf_err}", 500
    except Exception as e:
        print(f"Error during finalize_save: {e}")
        # Clean up flag file if it was created before error
        if os.path.exists(FLAG_FILE):
            try: os.remove(FLAG_FILE)
            except: pass
        return f"An internal error occurred: {e}", 500

@app.route("/shortcuts")
def shortcuts_page():
    conn = sqlite3.connect("shortcuts.db")
    cursor = conn.cursor()
    cursor.execute("SELECT gesture, command FROM shortcuts")
    records = cursor.fetchall()
    conn.close()

    shortcuts = []
    for gesture, command in records:
        filename = f"{gesture}.jpg"
        img_path = os.path.join("gesture_images", filename)

        # Clean path for HTML
        cleaned_filename = os.path.basename(img_path)  # removes all folder structure
        shortcuts.append((gesture, command, cleaned_filename))

    return render_template("shortcuts.html", shortcuts=shortcuts)

@app.route("/gesture_images/<filename>")
def get_image(filename):
    # Check if the requested file is the temporary file
    if filename == TEMP_IMAGE_FILE:
        # Serve from current working directory (where merged_vmouse saved it)
        print(f"Serving temporary image: {filename} from {os.getcwd()}")
        return send_from_directory(os.getcwd(), filename)
    else:
        # Serve from the UPLOAD_FOLDER as before
        print(f"Serving gesture image: {filename} from {UPLOAD_FOLDER}")
        return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
