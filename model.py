from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np


def train_and_check_model():
    train_dir = 'train'
    val_dir = 'val'
    test_dir = 'test'

    epochs = 5
    batch_size = 15
    nb_train_samples = 4311
    nb_validation_samples = 135
    nb_test_samples = 120


    # vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #
    # vgg19.trainable = False
    #
    # model = Sequential()
    # model.add(vgg19)
    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(6))
    # model.add(Activation('sigmoid'))

    model = load_model('fingers_neural.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=180, zoom_range=0.2)


    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size)

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size)

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save('fingers_neural.h5')

    plt.plot(history.history['accuracy'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history['val_accuracy'],
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()


    scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

    print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))


def predicting_model(name_img):
    model = load_model('fingers_neural.h5')

    img = image.load_img(name_img,
                         target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    result = model.predict(x)

    return result

