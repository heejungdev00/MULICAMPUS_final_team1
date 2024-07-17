import cv2
import mediapipe as mp
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 모델 및 토크나이저 로드
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# MediaPipe Pose 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 선택된 랜드마크 및 인덱스
SELECTED_LANDMARKS = ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
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
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    
    return frame, results

def get_angles(landmarks):
    angles = []
    
    # 오른쪽 고관절 각도
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    angle = calculate_angle(right_shoulder, right_hip, right_knee)
    angles.append(angle)
    
    # 왼쪽 고관절 각도
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    angle = calculate_angle(left_shoulder, left_hip, left_knee)
    angles.append(angle)
    
    # 오른쪽 무릎 각도
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    angle = calculate_angle(right_hip, right_knee, right_ankle)
    angles.append(angle)
    
    # 왼쪽 무릎 각도
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    angle = calculate_angle(left_hip, left_knee, left_ankle)
    angles.append(angle)
    
    return angles

def generate_feedback(angles):
    angles_text = ', '.join([f'{angle:.2f}' for angle in angles])
    input_text = f"관절 각도 차이: {angles_text}. 피드백:"
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
    feedback_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return feedback_text

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)  # 웹캠 캡처
cap_video = cv2.VideoCapture('your_video.mp4')  # 참조 비디오 파일

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened() and cap_video.isOpened():
        ret, frame = cap.read()
        ret_video, frame_video = cap_video.read()
        
        if not ret or not ret_video:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame, results = process_frame(frame, pose)
        frame_video, results_video = process_frame(frame_video, pose)
        
        if results.pose_landmarks and results_video.pose_landmarks:
            angles_webcam = get_angles(results.pose_landmarks.landmark)
            angles_video = get_angles(results_video.pose_landmarks.landmark)
            
            angle_differences = np.array(angles_webcam) - np.array(angles_video)
            
            # 각도 차이를 기반으로 피드백 생성
            feedback = generate_feedback(angle_differences)
            
            # 피드백을 화면에 표시
            joint_names = ["Right Hip", "Left Hip", "Right Knee", "Left Knee"]
            for i, (name, diff) in enumerate(zip(joint_names, angle_differences)):
                cv2.putText(frame, f"{name}: {diff:.2f}", 
                            (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, feedback, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            print(feedback)
        
        # 이미지 크기 맞추기
        height = min(frame.shape[0], frame_video.shape[0])
        width = min(frame.shape[1], frame_video.shape[1])
        
        frame_resized = cv2.resize(frame, (width, height))
        frame_video_resized = cv2.resize(frame_video, (width, height))
        
        # 두 영상을 나란히 표시
        combined_frame = np.hstack((frame_resized, frame_video_resized))
        cv2.imshow('Webcam and Video Comparison', combined_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cap_video.release()
cv2.destroyAllWindows()
