from ultralytics import YOLO
import cv2
import math
from flask import Flask, render_template, Response, request, url_for, redirect

app = Flask(__name__)

# Model ve sınıf isimleri
model1 = YOLO("C:/Users/PC/PycharmProjects/bitirme2/yolo-Weights/best.pt")
model2 = YOLO("C:/Users/PC/PycharmProjects/bitirme2/yolo-Weights/yolov8n.pt")

classNames1 = ['keyboard', 'laptop', 'monitor', 'mug', 'person', 'plant']
classNames2 =  [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

number_of_object=0

current_model = model1
classNames = classNames1

@app.route('/change_model', methods=['POST'])
def change_model():
    global current_model, classNames
    model_choice = request.form.get('model_choice')
    if model_choice == 'model1':
        current_model = model1
        classNames = classNames1
    else:
        current_model = model2
        classNames = classNames2
    return redirect(url_for('index'))



def detect_objects():
    # Kamera başlatma
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()  # Kameradan frame al
        if not success:
            break
        else:
            results = current_model(frame, stream=True)

            for r in results:
                boxes = r.boxes
                global number_of_object
                count = len(boxes)
                number_of_object = count

                for box in boxes:
                    # Kutu çiz
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # Sınıf adı
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # Nesne detayları
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(frame, (classNames[cls] + ' ' + str(confidence)+'***'+str(number_of_object)), org, font, fontScale, color,
                                thickness)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Frame'i client'a gönder


@app.route('/')
def index():
    global number_of_object
    object_count= number_of_object
    return render_template('index.html', count= object_count)  # HTML template


@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  # Kameradan gelen frame'leri client'a gönder


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
