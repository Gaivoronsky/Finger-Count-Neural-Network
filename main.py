from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2
from model import predicting_model, train_and_check_model
from write_dataset import write_dataset


def use_code():
    test_dir = 'pred'
    img_width, img_height = 224, 224
    datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height))

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
            cv2.imwrite("pred\\1\\test" + ".jpg", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if i % 5 == 0:
                prediction = model.predict(test_generator)
                kol = str(prediction.argmax())
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

use_code()