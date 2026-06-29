import cv2
import dlib
from datetime import datetime
import time
from gaze_tracking import GazeTracking

# Initialize face detector, facial landmark detector, and facial expression classifier
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
gaze = GazeTracking()

# Initialize flag for displaying "EYES CLOSED" message
display_message = True

close_eyes = True
close_eyes_time_end = datetime.now()
close_eyes_time_start = datetime.now()
blinking_count = 0
total_time = None

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:

    # Read a frame from the video stream
    ret, frame = cap.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Close eyes"
        blinking_count += 1
        close_eyes = True
        total_time = (datetime.now() - close_eyes_time_start).total_seconds()
        close_eyes_time_start = datetime.now()

    elif gaze.is_right():
        text = "Looking right"

    elif gaze.is_left():
        text = "Looking left"

    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (70, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0),
                1)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (70, 110),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (70, 130),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Loop over each face in the frame
    for face in faces:
        cv2.putText(frame, f"{len(faces)} face(s) found", (70, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(frame, f"{blinking_count} blinking", (70, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(frame, f"interval between blinks {total_time}", (70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # Get the coordinates of the face rectangle
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

        # Получение координат контрольных точек и их построение на изображении
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # Exit the loop if the "q" key is pressed
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.putText(frame, 'Screenshot saved', (70, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite('screenshots.png', frame)
        time.sleep(3)

# Release the camera and close all windows

cap.release()
cv2.destroyAllWindows()
"""TRASH"""
# Define a function to calculate the eye aspect ratio
# def eye_aspect_ratio(eye):
#     # Calculate the distance between the vertical eye landmarks
#     A = euclidean_dist(eye[1], eye[5])
#     B = euclidean_dist(eye[2], eye[4])
#     # Calculate the distance between the horizontal eye landmarks
#     C = euclidean_dist(eye[0], eye[3])
#     # Calculate the eye aspect ratio
#     ear = (A + B) / (2.0 * C)
#     return ear

# Define a function to calculate the Euclidean distance between two points
# def euclidean_dist(pt1, pt2):
#     return np.sqrt(np.sum((pt1 - pt2)**2))
