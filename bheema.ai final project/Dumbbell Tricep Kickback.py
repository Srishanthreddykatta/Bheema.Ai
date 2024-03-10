import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    kickback_counter = 0
    stage = None  # Possible values: "down", "up"

    while cap.isOpened():
        ret, frame = cap.read()

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get the pose landmarks
        results = pose.process(color)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Get coordinates for the shoulder, elbow, and wrist
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Check the tricep kickback's stage
            if angle > 160:
                stage = "up"
            if angle < 100 and stage == 'up':  # When the arm is bent more than 80 degrees
                stage = "down"
                kickback_counter += 1

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Display kickback count
            cv2.putText(frame, f'Tricep Kickbacks: {kickback_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw the progress bar
        progress = 100 - (angle - 90) if angle < 190 and angle > 90 else 0  # Assuming 100% when the arm is fully extended and 0% when it's bent
        cv2.rectangle(frame, (10, 100), (110, 150), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 100), (10 + int(progress), 150), (255, 0, 0), -1)

        cv2.imshow('Dumbbell Tricep Kickback Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()