import cv2
from os import walk


def write_dataset(class_data):
    cap = cv2.VideoCapture(0)

    wavs = []
    for (_, _, filenames) in walk('train\\' + class_data):
        wavs.extend(filenames)
        break

    i = len(wavs)
    while(cap.isOpened()):
        i += 1
        ret, frame = cap.read()
        if ret==True:
            cv2.imwrite("train\\" + class_data + "\\" + str(i) + ".jpg", frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()