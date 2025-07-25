# AirMouse: AI Virtual Mouse with Custom Shortcuts
Control your PC with hand gestures - create your own shortcuts and automate everything without touching your mouse.

## Overview
AirMouse is a gesture-controlled virtual mouse system built with Python.
It combines real-time hand tracking (using MediaPipe and OpenCV) with gesture recognition to:

- Move the mouse

- Left-click, right-click, scroll

- Trigger your own custom commands (like opening apps, websites or files)

- All this — just by waving your hand in front of the webcam!

## Technologies Used

- Hand tracking :	MediaPipe
- Video capture : OpenCV	
- Mouse control :	PyAutoGUI
- Gesture storage	: SQLite
- Web UI :	Flask
 
## How It Works 

- Detects 21 hand landmarks in real time →
- Identifies which fingers are up →
- Matches live landmarks with saved gestures using Euclidean distance 
- If matched, executes the associated command 
- Custom gestures and commands can be added dynamically through a browser UI
- Mouse movement, click, right-click and scroll all work through hand gestures

## Custom Gestures Flow

Press 's' - Capture current hand pose as temp image

Browser form opens - Enter gesture name & command (URL/app path)

On save -

Temp image moves to gesture_images/

Gesture data saved in shortcuts.db

A flag file (resume_vmouse.flag) tells main app to resume

## Mouse Controls

- Gesture	Action
- Index finger up	Move cursor
- Index + middle fingers up	Left click / scroll
- Thumb + index pinch	Right click
- Custom gestures (any pose)	Trigger saved command

## Setup & Run

- Clone the repository  
    git clone https://github.com/saumya-singh-14/AirMouse-AI-Virtual-Mouse-with-Custom-Shortcuts.git  
    cd vmouse5

- (Optional) Create virtual environment with Python 3.9/3.10  
    python -m venv venv  
    .\venv\Scripts\activate        # Windows  
    source venv/bin/activate      # macOS/Linux

- Install dependencies  
    pip install -r requirements.txt

- Initialize database (run once)  
    python database.py

- Start the virtual mouse  
    python merged_vmouse.py
