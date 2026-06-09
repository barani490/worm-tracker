# C. Elegans Worm Tracker

Real-time behavioral tracking system for C. elegans using Raspberry Pi 4 + HQ Camera.

Built(ing) as a part of SURF Research at the Ryu Lab, University of Toronto (2026)

## GOALS
- At the end of SURF, having a fully functioning setup that helps with thermotaxis research, where it records live occurence of a C. elegans thermotaxis experiment, tracks worms, and seamlessly outputs relevant information and data for analysis

## WHAT IT DOES (so far)
- Captures live video from Raspberry Pi camera
- Detects "individual worms" (haven't tested on a live set up yet, only videos) using background subtraction and blob detection
- Tracks worm identities across frames using Hungarian algorithm assignment (making tracking of individual worms very much easier)
- Logs position data (x, y, timestamp) to CSV
- Visualizes trajectories with matplotlib (work in progress)
- Can lay down setup into C. elegans experiment and calibrate to the environment (work in progress)
- MORE TO COME

## HARDWARE
- Raspberry Pi 4 (8GB)
- Raspberry Pi HQ Camera + 12mm C-mount lens
- Raspberry Pi Camera Module 3

## Dependencies
- opencv-python
- numpy
- scipy
- picamera2
- pandas
- matplotlib
