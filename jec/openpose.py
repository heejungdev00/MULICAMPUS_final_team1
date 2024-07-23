import numpy as np
import cv2
import pydicom
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

def weighted_log_loss(y_true, y_pred):
    class_weights = tf.constant([1.0, 2.0, 4.0])
    y_true = tf.cast(y_true, y_pred.dtype)
    weights = tf.reduce_sum(y_true * class_weights, axis=-1)
    loss = tf.reduce_sum(y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()), axis=-1)
    weighted_loss = -weights * loss
    return tf.reduce_mean(weighted_loss)

# 모델 불러오기
model_path = "C:/Users/user/MULICAMPUS_final_team1/my_trained_model.h5"
loaded_model = load_model(model_path, custom_objects={'weighted_log_loss': weighted_log_loss})

def index(request):
    return render(request, 'pybo/index.html')

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_url = fs.url(filename)

        # DICOM 파일 읽기 및 전처리
        dicom_path = fs.path(filename)
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        image = cv2.resize(image, (224, 224))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
        images = [image]
        x = np.array(images)

        # 모델 예측
        predictions = loaded_model.predict(x)
        result = predictions[0].tolist()  # 예측 결과를 리스트로 변환

        # 클래스 이름
        class_names = ['Normal/Mild', 'Moderate', 'Severe']
        predicted_class = class_names[np.argmax(result)]

        # 확률을 퍼센트로 변환
        result_percentages = [round(prob * 100, 2) for prob in result]

        return render(request, 'pybo/result.html', {
            'normal_mild': result_percentages[0],
            'moderate': result_percentages[1],
            'severe': result_percentages[2],
            'predicted_class': predicted_class,
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'pybo/upload.html')

def result(request):
    return render(request, 'pybo/result.html')

def openpose(request):
    if request.method == 'POST' and request.FILES['video']:
        video = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        video_path = fs.path(filename)

        # OpenPose 처리
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        cap = cv2.VideoCapture(video_path)
        frame_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame_list.append(frame)

        cap.release()
        pose.close()

        # 처리된 영상을 저장
        output_video_path = video_path.replace('.mp4', '_output.mp4')
        height, width, layers = frame_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

        for frame in frame_list:
            video_out.write(frame)
        
        video_out.release()

        return render(request, 'pybo/openpose_result.html', {
            'uploaded_video_url': fs.url(filename),
            'output_video_url': fs.url(output_video_path)
        })
    return render(request, 'pybo/openpose.html')
