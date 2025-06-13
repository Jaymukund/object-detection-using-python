from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)
ai_enabled = True
fps = 16

model = YOLO('yolov5s.pt')
camera = cv2.VideoCapture(0)

def generate_frames():
    global ai_enabled, fps
    while True:
        start_time = time.time()
        success, frame = camera.read()
        if not success:
            break
        else:
            if ai_enabled:
                results = model(frame)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            time.sleep(max(0, (1.0 / fps) - (time.time() - start_time)))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_ai', methods=['POST'])
def toggle_ai():
    global ai_enabled
    ai_enabled = not ai_enabled
    return {'status': 'AI toggled', 'ai_enabled': ai_enabled}

@app.route('/set_fps', methods=['POST'])
def set_fps():
    global fps
    fps = int(request.form['fps'])
    return {'status': 'FPS updated', 'fps': fps}

if __name__ == "__main__":
    app.run(debug=True)
