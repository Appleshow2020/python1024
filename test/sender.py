from flask import Flask, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 기본 카메라

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # JPEG 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "카메라 스트리밍 서버 실행 중..."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
