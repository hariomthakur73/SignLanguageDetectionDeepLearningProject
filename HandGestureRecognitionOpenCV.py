import cv2
import numpy as np
import math

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not detected")
    exit()

while True:
    ret, img = cap.read()
    if not ret or img is None:
        print("Failed to capture frame")
        continue

    # Define region of interest
    cv2.rectangle(img, (400, 400), (50, 50), (0, 255, 0), 0)
    crop_img = img[50:400, 50:400]

    # Convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(grey, (35, 35), 0)
    
    # Apply threshold
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

    # Find contours
    contours, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("No contours detected")
        continue

    # Get largest contour
    cnt = max(contours, key=cv2.contourArea)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

    # Convex hull and defects
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    if defects is None:
        print("No defects detected")
        continue

    count_defects = 0

    # Draw contours
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # Calculate angles
        a = math.dist(end, start)
        b = math.dist(far, start)
        c = math.dist(end, far)
        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0, 0, 255], -1)

        cv2.line(crop_img, start, end, [0, 255, 0], 2)

    # Gesture recognition
    gestures = {
        1: "GESTURE ONE",
        2: "GESTURE TWO",
        3: "GESTURE THREE",
        4: "GESTURE FOUR"
    }
    text = gestures.get(count_defects, "Hello World!!!")

    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Display results
    cv2.imshow('Gesture', img)
    cv2.imshow('Contours', np.hstack((crop_img, thresh1)))

    # Exit condition
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
