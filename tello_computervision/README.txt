2019-02-24

prediction_test.py
: 사람을 인식하고, 박스로 표시.

Tello3.py
: 텔로와 연결하여 송출되는 영상에 Gray필터를 씌움.

prediction_test_GRAY.py
: 흑백이미지로 속도 향상을 취하려고 하였으나



Tello3_objectdetection.py
: Tello에서 송출되는 영상의 프레임을 버림으로써
속도 향상을 취함.



연산량을 줄이고 속도를 향상시키기위해서

1. 컬러(RGB) -> GRAY (255) , BINARY ..
    FPS 코드 사용해서, 속도 차이 기록 해둘것

2. PIL 에서 opencv로 변환 opencv에서 PIL로 변환하는 과정
연산량 많기 때문에, opencv로만 연산 가능한지 


3. 이미지 자체의 크기를 줄임(해상도 낮춤)
 
4.skip 으로 프레임을 버림.



: 간단한 명령어를 추가하여, 앞, 뒤 옆, 위 아래로 이동할수 있게 함.


내일은 따라가는 것을 만들어보려고 함.