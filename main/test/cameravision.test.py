import cv2
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        print("error")
    cv2.imshow("test",frame)