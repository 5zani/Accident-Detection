# your_object_detection_code.py

import cv2
import numpy as np

def process_video(video_path):
    # Load pre-trained MobileNet SSD model
    net = cv2.dnn.readNetFromCaffe(
        r"C:/Users/User/Desktop/OpenCV/myvenv/Scripts/deploy (1).prototxt",
        r"C:/Users/User/Desktop/OpenCV/myvenv/Scripts/mobilenet_iter_73000 (1).caffemodel"
    )

    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    CONFIDENCE_THRESHOLD = 0.5

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detect_accident(frame, net, CLASSES, CONFIDENCE_THRESHOLD)
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def detect_accident(frame, net, CLASSES, CONFIDENCE_THRESHOLD):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    vehicle_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] in ["car", "bus", "motorbike"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                vehicle_boxes.append((startX, startY, endX, endY))

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    if len(vehicle_boxes) > 1:
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                (x1, y1, x2, y2) = vehicle_boxes[i]
                (x3, y3, x4, y4) = vehicle_boxes[j]

                if not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1):
                    cv2.putText(frame, "Accident Detected!", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    break

    return frame
