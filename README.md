# SpineGuardians

## 웹사이트 코드
[웹사이트 제작코드](https://drive.google.com/drive/folders/1DxJaCmV9aMphTQpw9SjCNQIPEKXdRtvW?usp=sharing
)
## 구성원(역할)
**추희정**
```
- 요추 퇴행성 분류 모델 총괄

- 웹사이트 자연어처리모델 연동

- 바로가기 연동
```
**전은채**
```
-웹사이트 총괄
```
**조한영**
```
-요추 퇴행성 분류 샘플모델 모델링

-자세교정모델 모델링,총괄

-자연어처리모델 모델링,총괄

```

## 프로젝트 주제
요추퇴행성 분류모델제작 & 웹사이트제작

## 기획의도
 요추 퇴행성 질환은 고령화 사회와 현대인의 생활 습관으로 인해 발생 빈도가 증가하고 있는 중요한 건강 문제이다. 이 프로젝트는 요추 퇴행성 질환 진단에 도움을 주는 인공지능(ai) 기반 분류 모델을 만들고 이를 사용자 친화적인 웹 플랫폼으로 제공함으로써, 의료 전문가에게 도움을 주고자 한다.

## 주요분석 내용
1. 데이터 이해와 전처리 과정
   - 의료 이미지 데이터 이해
   - 이미지 정규화, 데이터 라벨링 등의 전처리   
2. 모델 개발
   - CNN을 사용하여 의료 영상 분류 모델을 만듦
   - s2s모델을 활용하여 피드백 모델 개발
3. 모델 평가& 결과 시각화
4. 만들어진 모델 활용
   - 모델을 이용하여 요추 퇴행성 질환 예측
   - 이를 바탕으로 질환에 따른 약물치료, 치료방법 추천
   - openpose 알고리즘을 적용하여 환자의 자세를 분석하고,
 이를 바탕으로 환자 자세교정 및 운동법 추천
   - 운동법및 자세교정을 자연어 처리모델을 활용하여 직관적으로 이해하기 쉽게 설명.
   - tts를 활용하여 사용자가 음성으로 피드백을 들을 수 있게 생성

## 기대효과
요추퇴행성 통증 정도와 증상을 정확히 분류하고 이에 따른 알맞은 대처 및 운동방법을 제시하여 향후 관리에 도움이 될 수 있다.

## 기능
- kaggle RSNA 2024 Lumbar Spine Degenerative 데이터를 이용 

- python의 pandas, sklearn 모듈을  사용하여  데이터 전처리와 분석

- cnn을 활용한 딥러닝 모델링을 하여 각 환자의 통증정도를 분류하여 이에 따른 운동 방법과 자세를 추천

- openpose 알고리즘을 활용해서 운동 및 자세를 인식하고 교정
    - 환자가 더 직관적으로 알아들을 수 있게 자연어 처리를 활용하여 텍스트를 생성하고 이를 음성으로 말해주는 api를 연동하여 운동을 하며 즉각 피드백


## 사용기술(툴)
- python, pandas, sklearn, tensorflow
- cnn
- openpose 알고리즘(mediapipe)
- s2s
- django
- jupythernotebook, 구글코랩
- 협업툴
  - github
  - googledrive
  - 
## 구현방법
**최종 보고서 참고**
![시연영상](report/시연영상.gif)

## 데이터 출처
- [캐글-요추퇴행성분류](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview)
- 자연어처리 모델은 직접 제작