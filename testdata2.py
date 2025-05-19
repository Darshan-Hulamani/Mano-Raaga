import os
# Suppress TensorFlow debugging info, warnings, and error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

# Optional: to suppress all console output from TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings("ignore")  # Suppress Python warnings

import cv2
import numpy as np
from keras.models import load_model

def recognize(filename):
    # Load the model
    model = load_model('load/emotion_model_enhanced.h5')
    detected_emotion = []

    # Recompile the model to set metrics and optimizer state
    model.compile(optimizer='adam', loss='your_loss_function', metrics=['accuracy'])

    # Load the face detection model
    faceDetect = cv2.CascadeClassifier('load/haarcascade_frontalface_default.xml')

    # Define labels for emotions
    labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    # Read the image
    frame = cv2.imread(f"captured_images/{filename}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    # Process each detected face
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        
        # Predict the emotion
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        # # Draw rectangles and labels on the image
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        # cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        # cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        detected_emotion.append(labels_dict[label])
    # # Display the result
    # cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.save("emotion.npy", np.array(detected_emotion[-1]))
    # for emotion in detected_emotion:
    #     print(emotion)
    # print("_______________")
    emo = np.load("emotion.npy")
    print(emo)
    return emo