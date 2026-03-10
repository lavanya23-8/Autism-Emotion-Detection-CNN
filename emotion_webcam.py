import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels
labels = ["anger","fear","joy","natural","sadness","surprise"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize
    img = cv2.resize(gray, (48,48))

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1,48,48,1)

    # Predict
    prediction = model.predict(img)
    emotion = labels[np.argmax(prediction)]

    # Show result on screen
    cv2.putText(frame, emotion, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()