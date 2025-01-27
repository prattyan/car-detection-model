import cv2
import numpy as np


def load_yolo():
    
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.weights")
    with open("D:\Programs\Python\Car Number Detection\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

net, classes, output_layers = load_yolo()


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (600, 400))  # Optional resizing
    return img

def load_video(video_path):
    return cv2.VideoCapture(video_path)


def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs, width, height


def get_box_dimensions(outputs, width, height, confidence_threshold=0.5):
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def draw_labels(img, boxes, confidences, class_ids, classes, car_label="car", confidence_threshold=0.5):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == car_label:
                confidence = confidences[i]
                color = (0, 255, 0)  # Green box for cars
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} {round(confidence * 100, 2)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def detect_from_video(video_path, net, classes, output_layers):
    cap = load_video(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs, width, height = detect_objects(frame, net, output_layers)
        boxes, confidences, class_ids = get_box_dimensions(outputs, width, height)
        draw_labels(frame, boxes, confidences, class_ids, classes)

        
        cv2.imshow('Car Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_from_image(img_path, net, classes, output_layers):
    img = load_image(img_path)
    outputs, width, height = detect_objects(img, net, output_layers)
    boxes, confidences, class_ids = get_box_dimensions(outputs, width, height)
    draw_labels(img, boxes, confidences, class_ids, classes)

    
    cv2.imshow("Car Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    net, classes, output_layers = load_yolo()

   

    video_path = 'road_footage.mp4' 
