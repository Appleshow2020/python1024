# ui/web_dashboard.py
from flask import Flask, render_template, Response
import cv2
import time
# 웹 대시보드를 독립적으로 실행하기 위한 임시 데이터 소스
# 실제로는 Redis나 RabbitMQ 같은 메시지 브로커를 사용하는 것이 더 안정적입니다.
# 여기서는 시연을 위해 파일 기반의 간단한 데이터 교환을 가정합니다.
# (또는 main 프로세스와 별도로 실행되어야 합니다.)

# 이 파일은 현재 구조에서는 improved_main.py와 직접 연동되지 않습니다.
# 만약 연동하려면, 데이터 브로커를 통해 프레임을 받아오는 로직이 필요합니다.
# 아래는 Flask를 이용한 웹 스트리밍의 기본 예시 코드입니다.

app = Flask(__name__)

def generate_frames():
    """카메라에서 프레임을 가져와 JPEG 형식으로 인코딩하여 스트리밍합니다."""
    camera = cv2.VideoCapture(0) # 웹 대시보드용 독립 카메라
    if not camera.isOpened():
        print("웹 대시보드용 카메라를 열 수 없습니다.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # (여기에 detection 결과를 프레임에 그리는 로직 추가 가능)
            
            # 프레임을 JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            
            # HTTP 스트리밍 형식에 맞춰 yield
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/')
def index():
    """대시보드 HTML 페이지를 렌더링합니다."""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """비디오 스트리밍 경로"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # config.json에서 호스트, 포트 정보를 읽어오도록 수정할 수 있습니다.
    app.run(host='0.0.0.0', port=8080, debug=True)