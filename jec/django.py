# jec/pybo/views.py
from django.shortcuts import render
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose

SELECTED_LANDMARKS = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
    'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
    'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
    'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

SELECTED_LANDMARK_INDICES = [mp_pose.PoseLandmark[landmark].value for landmark in SELECTED_LANDMARKS]

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def process_frame(frame, pose):
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        connections_to_draw = [
            connection for connection in mp_pose.POSE_CONNECTIONS
            if connection[0] in SELECTED_LANDMARK_INDICES and connection[1] in SELECTED_LANDMARK_INDICES
        ]
        
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, connections_to_draw,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    return frame, results

def get_angles(landmarks):
    angles = []
    
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    angles.append(angle)

    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angles.append(angle)
    
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    angle = calculate_angle(right_hip, right_knee, right_ankle)
    angles.append(angle)

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    angle = calculate_angle(left_hip, left_knee, left_ankle)
    angles.append(angle)

    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    angle = calculate_angle(right_knee, right_ankle, right_foot_index)
    angles.append(angle)

    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    angle = calculate_angle(left_knee, left_ankle, left_foot_index)
    angles.append(angle)

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    angle = calculate_angle(right_shoulder, right_hip, right_knee)
    angles.append(angle)

    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    angle = calculate_angle(left_shoulder, left_hip, left_knee)
    angles.append(angle)
    
    return angles

def process_video(request):
    if request.method == 'POST' and request.FILES['video_file']:
        video_file = request.FILES['video_file']
        cap = cv2.VideoCapture(video_file.temporary_file_path())
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        angles_per_frame = []
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame, results = process_frame(frame, pose)
                if results.pose_landmarks:
                    angles = get_angles(results.pose_landmarks.landmark)
                    angles_per_frame.append(angles)
        
        cap.release()

        context = {
            'angles': angles_per_frame,
        }
        
        return render(request, 'pybo/result.html', context)
    
    return render(request, 'pybo/upload.html')








<!-- upload.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Upload Video</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            font-family: Arial, sans-serif;
        }
        .upload-form {
            text-align: center;
        }
        .upload-form input[type="file"] {
            margin: 20px 0;
        }
        footer {
            position: absolute;
            bottom: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background-color: #f1f1f1;
            box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <div class="upload-form">
        <h1>Upload Video to Analyze Pose</h1>
        <form method="post" enctype="multipart/form-data" action="{% url 'process_video' %}">
            {% csrf_token %}
            <input type="file" name="video_file" accept="video/*">
            <button type="submit">Upload</button>
        </form>
    </div>
    <footer>
        <div>© 2024 스핀가디언즈</div>
        <div><a href="https://github.com/your-github-repo" target="_blank">GitHub</a></div>
    </footer>
</body>
</html>
