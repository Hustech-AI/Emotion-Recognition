import cv2
from keras.models import model_from_json
import numpy as np

# Load the trained model
json_file = open("C:/Users/Hussnain Khalid/Machine Learning Projects/AI Datayard/FER 13 Self Project/emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("C:/Users/Hussnain Khalid/Machine Learning Projects/AI Datayard/FER 13 Self Project/emotiondetector.h5")

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the face cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Iterate through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face image to match the model input size
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension

        # Normalize the face image
        face_roi = face_roi / 255.0

        # Reshape for model prediction
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Predict emotion
        emotion_probabilities = model.predict(face_roi)
        predicted_label = label[np.argmax(emotion_probabilities)]

        # Display the emotion prediction on the frame
        cv2.putText(frame, f'Emotion: {predicted_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
