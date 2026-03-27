============================================================
           HAND SIGN LANGUAGE TO TEXT PROJECT
============================================================

An interactive Computer Vision project that uses MediaPipe 
and OpenCV to translate hand gestures into text in real-time.

------------------------------------------------------------
1.  PROJECT STRUCTURE
------------------------------------------------------------
* app.py                 - The main application (Live Translator).
* collect_data.py        - Tool to record hand data for new signs.
* train.py               - Trains the AI model (model.pkl).
* capture_references.py  - Captures hand photos for the UI sidebar.
* hand_landmarker.task   - MediaPipe's hand tracking model.
* landmarks.csv          - The database of recorded hand points.
* model.pkl              - The trained brain of the project.

------------------------------------------------------------
2.  ONE-TIME SETUP
------------------------------------------------------------
Ensure you have Python installed, then run this command in 
your terminal to install the necessary libraries:

pip install opencv-python mediapipe scikit-learn numpy

------------------------------------------------------------
3.  HOW TO RUN
------------------------------------------------------------
To start the live translation app, simply run:

python app.py

------------------------------------------------------------
4.  HOW TO ADD NEW SIGNS (OPTIONAL)
------------------------------------------------------------
If you want to train the model on your own hand gestures:

Step 1: Run 'python collect_data.py'
        Follow the prompts to record 200 samples per letter.
        
Step 2: Run 'python train.py'
        This updates 'model.pkl' with your new data.
        
Step 3: Run 'python capture_references.py'
        This updates the visual guide on the left of the app.

------------------------------------------------------------
5.  CONTROLS & SHORTCUTS
------------------------------------------------------------
While the app is running:

[SPACE]     - Add a space to your sentence.
[BACKSPACE] - Delete the last character.
[Q] or [ESC]- Quit the application.

* MOUSE CLICKS: You can also click the on-screen buttons 
  (Clear, Save, Quit) using your mouse.

------------------------------------------------------------
6.  TROUBLESHOOTING
------------------------------------------------------------
- No Camera: Ensure no other app (like Zoom/Teams) is 
  using your webcam.
- Accuracy: Ensure your hand is well-lit and clearly 
  visible to the camera.
- Missing Files: If 'model.pkl' is missing, you must 
  run 'train.py' before starting 'app.py'.

============================================================