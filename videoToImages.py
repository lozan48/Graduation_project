import cv2
import os
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf

# Ensure TensorFlow uses GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        print("TensorFlow is using GPU.")
    except RuntimeError as e:
        print(e)

# Parameters
video_path = 'test.mp4'  # Path to the video file
output_dir = 'extracted_faces'         # Directory to save the extracted faces
frame_interval = 5                     # Process every 5th frame to speed up

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the MTCNN face detector
detector = MTCNN()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
face_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process every nth frame (frame_interval)
    if frame_count % frame_interval != 0:
        continue

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        # Extract the face from the frame
        face_img = frame[y:y+h, x:x+w]

        # Save the face image
        face_filename = os.path.join(output_dir, f'face_{face_count}.png')
        cv2.imwrite(face_filename, face_img)
        face_count += 1

    print(f'Processed frame {frame_count}, found {len(faces)} faces.')

# Release the video capture object
cap.release()

print(f'Extraction completed. {face_count} faces were saved to "{output_dir}".')
