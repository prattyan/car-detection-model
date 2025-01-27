import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained model
model = tf.saved_model.load('path/to/saved_model')

def detect_cars(image_path):
    # Read the image
    image = cv2.imread(image_path)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = model(input_tensor)

    # Process the results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() 
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Filter for car detections (assuming class ID 3 is for cars)
    car_indices = np.where(detections['detection_classes'] == 3)[0]
    car_boxes = detections['detection_boxes'][car_indices]
    car_scores = detections['detection_scores'][car_indices]

    # Draw bounding boxes for cars
    for box, score in zip(car_boxes, car_scores):
        if score > 0.5:  # Confidence threshold
            h, w, _ = image.shape
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f'Car: {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save or display the result
    cv2.imwrite('output.jpg', image)
    cv2.imshow('Car Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
detect_cars('path/to/your/image.jpg')