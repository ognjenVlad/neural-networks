import glob
import numpy as np
import matplotlib.pyplot as plot
import cv2
from datetime import date
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
import os


code = {'mountain': 0, 'street': 1, 'glacier': 2, 'buildings': 3, 'sea': 4, 'forest': 5}


def load_picture_paths(path):
    ret_value = {}
    for folder in os.listdir(path):
        ret_value[folder] = glob.glob(pathname=str(path + folder + '/*.jpg'))
    return ret_value


def load_and_resize(pictures, size=150):
    resized_pictures = {}
    for key in pictures.keys():
        current_array = []
        for picture in pictures[key]:
            image = cv2.imread(picture)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size))
            current_array.append(list(image))
        resized_pictures[key] = current_array
    return resized_pictures


def analyze_pictures(label, pictures):
    print('~'*20)
    print('Analyzing data from group > ' + label + ' <')
    for key in pictures.keys():
        print('Group > {:12s} < contains > {:7s} < images'.format(key, str(len(pictures[key]))))
    print('~' * 20)


def remove_different(pictures):
    total_pictures = sum([len(pictures[key]) for key in pictures.keys()])
    print('~' * 20)
    print('Removing all non (150,150) images..')
    print('Number of images before removal: ' + str(total_pictures))
    for key in pictures.keys():
        to_remove = []
        for file in pictures[key]:
            image = plot.imread(file)
            if image.shape != (150, 150, 3):
                to_remove.append(file)
        pictures[key] = [item for item in pictures[key] if item not in to_remove]
    total_pictures = sum([len(pictures[key]) for key in pictures.keys()])
    print('Number of images after removal: ' + str(total_pictures))
    print('~' * 20)


def create_XY(pictures):
    X_data = []
    Y_data = []
    for key in pictures.keys():
        for picture in pictures[key]:
            X_data.append(picture)
            Y_data.append(code[key])
    return np.array(X_data), np.array(Y_data)

if __name__ == '__main__':
    train_path = './data/seg_train/'
    test_path = './data/seg_test/'
    pred_path = './data/seg_pred/'

    train_pics = load_picture_paths(train_path)
    test_pics = load_picture_paths(test_path)

    analyze_pictures('train', train_pics)
    analyze_pictures('test', test_pics)

    remove_different(train_pics)
    remove_different(test_pics)

    train_pics = load_and_resize(train_pics)
    test_pics = load_and_resize(test_pics)

    X_train, Y_train = create_XY(train_pics)
    X_test, Y_test = create_XY(test_pics)

    # print('{} , {} , {} , {}'.format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))

    cnn_model = keras.models.Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(5, 5),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(5, 5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(rate=0.5),
        Dense(6, activation='softmax'),
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Model overview:')
    print(cnn_model)

    # 0.53 accuracy for batch size 64 ( 32 default - test it )
    cnn_model.fit(X_train, Y_train, batch_size=32, epochs=35, validation_split = 0.2)

    _, accuracy = cnn_model.evaluate(X_test, Y_test)

    print('Model accuracy on test > {} <'.format(accuracy))

    date_str = date.today().strftime("%d-%m-%Y")
    model_file = 'models/cnn' + date_str
    cnn_model.save(model_file)
