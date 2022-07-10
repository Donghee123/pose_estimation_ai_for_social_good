# pose_estimation_ai_for_social_good
Improve Our Posture to Save Our Future

앉은 자세 추정 프로젝트 코드 입니다.

2스테이지로 추정을 하는 구조입니다.

1 스테이지 : 
input : 3 channel image 
output : skeleton 좌표 추출  shape (1 x 132)

2 스테이지 : 
input : skeleton 좌표 데이터 (1 x 132)
output : predictions (good, left, right, forward head)
