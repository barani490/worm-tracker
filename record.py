from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import time

#configuring recording parameters
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720)})
picam2.configure(config)

#establishing stuff
encoder = H264Encoder()
output = FileOutput("recording.h264")

picam2.start_recording(encoder, output) #recording the HQ camera input
picam2.set_controls({ #frame would refresh which would jitter the camera, this sort of fixed it
    "AeEnable": False,
    "ExposureTime": 10000,
    "AnalogueGain": 2.0
})
time.sleep(1)
print("Recording... press Ctrl+C to stop")

#for stopping the recording
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop_recording()
    print("Saved to recording.h264")
