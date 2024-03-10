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
    curl_counter = 0
    stage = None  # Possible values: "up", "down"

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

            # Check the curl's stage
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == 'down':
                stage = "up"
                curl_counter += 1

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Display curl count
            cv2.putText(frame, f'Reverse Curls: {curl_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw the progress bar
        progress = 100 - angle if angle < 100 else 0  # Assuming 100% when the arm is fully down and 0% when it's fully up
        cv2.rectangle(frame, (10, 100), (110, 150), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 100), (10 + int(progress), 150), (255, 0, 0), -1)

        cv2.imshow('Reverse Curls Pose Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()