import cv2
import numpy as np
import pytesseract

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 10, 200)
    return edged

def find_contours(processed_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    return contours

def get_license_plate(contours, image):
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4:
            license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            plate = image[y:y+h, x:x+w]
            return plate, license_plate
    return None, None

def ocr_license_plate(plate):
    config = ('-l eng --oem 1 --psm 3')
    text = pytesseract.image_to_string(plate, config=config)
    return text.strip()

def detect_license_plate(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    contours = find_contours(processed_image)
    plate, license_plate = get_license_plate(contours, image)
    
    if plate is not None:
        cv2.drawContours(image, [license_plate], -1, (0, 255, 0), 3)
        text = ocr_license_plate(plate)
        print(f"Detected License Plate: {text}")
        cv2.imshow("License Plate Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No license plate found")

# Usage
detect_license_plate('car_image.jpg')