import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import save_model, load_model

# ------------ Assignment 2 imports ---------------
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from scipy.stats import ttest_rel


def load_imgs(path, folders):
    imgs = []
    labels = []
    n_imgs = 0
    for c in folders:
        # iterate over all the files in the folder
        for f in os.listdir(os.path.join(path, c)):
            if not f.endswith('.jpg'):
                continue
            # load the image (here you might want to resize the img to save memory)
            im = Image.open(os.path.join(path, c, f)).copy()
            imgs.append(im)
            labels.append(c)
        print('Loaded {} images of class {}'.format(len(imgs) - n_imgs, c))
        n_imgs = len(imgs)
    print('Loaded {} images total.'.format(n_imgs))
    return imgs, labels


def plot_sample(imgs, labels, nrows=4, ncols=4, resize=None):
    # create a grid of images
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    # take a random sample of images
    indices = np.random.choice(len(imgs), size=nrows * ncols, replace=False)
    for ax, idx in zip(axs.reshape(-1), indices):
        ax.axis('off')
        # sample an image
        ax.set_title(labels[idx])
        im = imgs[idx]
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        if resize is not None:
            im = im.resize(resize)
        ax.imshow(im, cmap='gray')


# map class -> idx
label_to_idx = {
    'CHEETAH': 0,
    'OCELOT': 1,
    'SNOW LEOPARD': 2,
    'CARACAL': 3,
    'LIONS': 4,
    'PUMA': 5,
    'TIGER': 6
}

idx_to_label = {
    0: 'CHEETAH',
    1: 'OCELOT',
    2: 'SNOW LEOPARD',
    3: 'CARACAL',
    4: 'LIONS',
    5: 'PUMA',
    6: 'TIGER'
}


def make_dataset(imgs, labels, label_map, img_size):
    x = []
    y = []
    n_classes = len(list(label_map.keys()))
    for im, l in zip(imgs, labels):
        # preprocess img
        x_i = im.resize(img_size)
        x_i = np.asarray(x_i)

        # encode label
        y_i = np.zeros(n_classes)
        y_i[label_map[l]] = 1.

        x.append(x_i)
        y.append(y_i)
    return np.array(x).astype('float32'), np.array(y)


def save_keras_model(model, filename):
    """
    Saves a Keras model to disk.
    Example of usage:

    >>> model = Sequential()
    >>> model.add(Dense(...))
    >>> model.compile(...)
    >>> model.fit(...)
    >>> save_keras_model(model, 'my_model.h5')

    :param model: the model to save;
    :param filename: string, path to the file in which to store the model.
    :return: the model.
    """
    save_model(model, filename)


def load_keras_model(filename):
    """
    Loads a compiled Keras model saved with models.save_model.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = load_model(filename)
    return model


# ------------ Assignment 2 Utils ---------------
def plot_history(history):
    """
    Plots the history of training of a model
    :param history: the history of the training procedure.
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def create_FFNN(input_shape):
    """
    Creates a Feed Forward NN model.
    :return: the created FFNN model.
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_cnn():
    """
    Creates a Convolutional NN model.
    :return: the created CNN model.
    """
    conv_model = Sequential()
    # Convolutional layers
    conv_model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(224, 224, 3)))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))

    conv_model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))

    conv_model.add(Conv2D(filters=128, kernel_size=3, strides=1, activation='relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Classifier
    conv_model.add(Flatten())
    conv_model.add(Dense(256, activation='relu'))
    conv_model.add(Dense(128, activation='relu'))
    conv_model.add(Dense(7, activation='softmax'))
    conv_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    return conv_model


def perform_paired_T_Test(model1_predictions, model2_predictions, y_test):
    """
    Performs a paired T-Test on the predictions of two models.
    :param model1_predictions:  predictions of the first model.
    :param model2_predictions:  predictions of the second model.
    :param y_test:              targets in the test set.
    :return:                    T and p-value of the test.
    """
    predicted_labels_model1 = np.argmax(model1_predictions, axis=1)
    predicted_labels_model1 = np.eye(7)[predicted_labels_model1]
    e_model2 = (predicted_labels_model1[:] == y_test[:] )\
        .all(axis=1)\
        .astype(int)

    predicted_labels_model2 = np.argmax(model2_predictions, axis=1)
    predicted_labels_model2 = np.eye(7)[predicted_labels_model2]
    e_model1 = (predicted_labels_model2[:] == y_test[:])\
        .all(axis=1)\
        .astype(int)

    return ttest_rel(e_model1, e_model2)
