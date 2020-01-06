from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from model import predicting_model, train_and_check_model
from write_dataset import write_dataset
import numpy as np


def use_code():
    model = load_model('fingers_neural.h5')

    cap = cv2.VideoCapture(0)

    i = 0
    kol = ''
    while(cap.isOpened()):
        i += 1
        ret, frame = cap.read()
        cv2.putText(frame, f"Fingers = {kol}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 6)
        if ret == True:

            cv2.imshow('frame', frame)
            cv2.imwrite("pred.jpg", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if i % 5 == 0:
                img = image.load_img('pred.jpg',
                                     target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                prediction = model.predict(x)
                kol = str(prediction.argmax())
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

use_code()
