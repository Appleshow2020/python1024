from typing import Dict
import cv2
import time
camera_count: int = int(input("Camera Count:"))
camera_indices: Dict[int, int] = {}
reali = 0
for i in range(10):  # 0~10번 장치 탐색
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"[{time.strftime('%X')}] [INFO] Camera found at index {i}")
        cap.release()
        camera_indices[reali] = i
        reali+=1

def show_multiple_cameras(indices):
    caps = []
    
    # 카메라 초기화
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Camera {idx} 열 수 없음.")
        else:
            caps.append((idx, cap))

    if not caps:
        print("카메라 없음.")
        return
    
    while True:
        for idx, cap in caps:
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {idx}에서 프레임 읽기 실패.")
                continue
            cv2.imshow(f"Camera {idx}", frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    for _, cap in caps:
        cap.release()
show_multiple_cameras(camera_indices.values())