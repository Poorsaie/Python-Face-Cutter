import cv2

# image path
image_path = "face.jpg"

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Check if any faces were detected
if len(faces) == 0:
    print("No face detected.")
else:
    # Assuming only one face is detected, take the first face
    (x, y, w, h) = faces[0]

    # Calculate the coordinates for the four halves of the face
    top_left = (x, y, w // 2, h // 2)
    top_right = (x + w // 2, y, w // 2, h // 2)
    bottom_left = (x, y + h // 2, w // 2, h // 2)
    bottom_right = (x + w // 2, y + h // 2, w // 2, h // 2)

    # Draw numbers on each part of the face
    cv2.putText(image, '1', (top_left[0] + 20, top_left[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, '2', (top_right[0] + 20, top_right[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, '3', (bottom_left[0] + 20, bottom_left[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, '4', (bottom_right[0] + 20, bottom_right[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Crop and display the four halves of the face
    cv2.imshow("Top Left (1)", image[top_left[1]:top_left[1] + top_left[3], top_left[0]:top_left[0] + top_left[2]])
    cv2.imshow("Top Right (2)", image[top_right[1]:top_right[1] + top_right[3], top_right[0]:top_right[0] + top_right[2]])
    cv2.imshow("Bottom Left (3)", image[bottom_left[1]:bottom_left[1] + bottom_left[3], bottom_left[0]:bottom_left[0] + bottom_left[2]])
    cv2.imshow("Bottom Right (4)", image[bottom_right[1]:bottom_right[1] + bottom_right[3], bottom_right[0]:bottom_right[0] + bottom_right[2]])

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
