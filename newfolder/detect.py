import cv2
import numpy as np
from picamera2 import Picamera2 #just need this for now
import csv
import time


class Worm:

    id: int
    positions: list[int]
    state: str
    missing_frames: int

    def __init__(self, worm_id, position):
        self.id = worm_id
        self.positions = [position]
        self.state = "TRACKED"
        self.missing_frames = 0

    def update(self, position):
        self.positions.append(position)
        self.missing_frames = 0
        self.state = "TRACKED"

    def mark_missing(self):
        self.missing_frames += 1
        if self.missing_frames > 30:  #need to play around with the frame refresh more
            self.state = "LOST"

    def last_position(self):
        return self.positions[-1]


def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def assign_worms(worms, centroids, max_distance=50):
    if not worms or not centroids:
        return {}, list(range(len(centroids)))

    #this part had a lot of help from AI
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    worm_positions = [w.last_position() for w in worms]
    cost_matrix = cdist(worm_positions, centroids)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = {}
    assigned_centroids = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_distance:
            assignments[r] = c
            assigned_centroids.add(c)

    unassigned = [i for i in range(len(centroids)) if i not in assigned_centroids]
    return assignments, unassigned


#initializing the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)}) #4608 X 2592 (12Mpx)
picam2.configure(config)
picam2.start()

#need to play around with camera settings more

#have to capture clean background to not have residue
#will change this for in the lab
print("Capturing background in 3 seconds - remove anything from view")
time.sleep(3)
bg_frame = picam2.capture_array()
background = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
background = cv2.normalize(background, None, 0, 255, cv2.NORM_MINMAX)
background = cv2.GaussianBlur(background, (5, 5), 0)
print("Background captured. Place object in view now.") #will change in lab
time.sleep(2)

#csv log
csv_file = open("worm_tracks.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["timestamp", "worm_id", "x", "y", "state"])

worms = []
next_id = 0
frame_count = 0

print("Tracking started. Press Ctrl+C to stop.")

#this is the actual chunk of the code
#had a ton of help and need to work on it more
try:
    while True:
        frame = picam2.capture_array()

        #this part converts and normalizes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        #this part subtracts static background
        diff = cv2.absdiff(gray, background)
        _, fg_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        #cleaning up the noise
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        #for finding contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #after everything, we filter by size and get the centroids of the worms
        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 2000:
                c = get_centroid(cnt)
                if c:
                    centroids.append(c)

        #if any worms exist, assign centroid
        assignments, unassigned = assign_worms(worms, centroids)

        #updating the assigned worms
        for worm_idx, centroid_idx in assignments.items():
            worms[worm_idx].update(centroids[centroid_idx])

        #the unassigned worms that are left are keyed as missing
        assigned_worm_indices = set(assignments.keys())
        for i, worm in enumerate(worms):
            if i not in assigned_worm_indices:
                worm.mark_missing()

        #creating new worms for worms that are unassigned
        for idx in unassigned[:5]:
            worms.append(Worm(next_id, centroids[idx]))
            next_id += 1

        #if worms are gone, we remove them
        worms = [w for w in worms if w.state != "LOST"]

        #log everything to csv
        timestamp = time.time()
        for worm in worms:
            pos = worm.last_position()
            writer.writerow([timestamp, worm.id, pos[0], pos[1], worm.state])

        frame_count += 1
        ids = [w.id for w in worms]
        print(f"Frame {frame_count}: {len(worms)} tracked. IDs: {ids}")

except KeyboardInterrupt:
    print(f"Stopped. Tracked {next_id} worms total.")
    csv_file.close()
    picam2.stop()
