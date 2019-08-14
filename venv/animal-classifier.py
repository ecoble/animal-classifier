# Allows division to return a float
from __future__ import division

# Allows access to the file system
import os

# Provides an API for scientific computing
import numpy as np

# Allows use to timestamp the training run
from datetime import datetime

# Allows us to render images and plot data
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import math
import matplotlib.pyplot as plt

# Machine learning framework that provides an abstract API on top of Tensorflow
import keras
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import optimizers

import tensorflow as tf


def get_class_labels(dir):
    """
    Gets the name of each sub-directory in the given directory.

    dir: Directory to search.
    return: An array of the names of the sub-directories in dir.
    """

    # Get all sub-directories in this directory
    classes = os.listdir(dir)

    return classes


def get_class_images(classes, dir):
    """
    Gets the paths of all images in each directory.

    classes: Name of each class.
    dir: Directory to search.
    return: A 2d array of paths organized by class name.
    """

    # Create an array to hold the image paths of each class
    class_paths = []

    # Create image paths of each class
    for label in classes:

        # Create an array to hold the image paths of this class (label)
        image_paths = np.array([])

        # Create the path of this class
        class_path = os.path.join(dir, label)

        # Get all images in this directory
        images = os.listdir(class_path)

        # Create the path of each images in this class
        for image in images:
            # Create the path of this image
            image_path = os.path.join(class_path, image)

            # Add the image path to the image paths array
            image_paths = np.append(image_paths, image_path)

        # Add the image paths to the class paths array
        class_paths.append(image_paths)

    return class_paths


def predict(batch_size, image_paths, model):
    """
    Makes predictions with the model

    batch_size: number of predictions to make
    image_paths: paths to images
    model: image classifier model
    return: resulting predictions
    """

    images_arr = []

    # load images
    for image_path in image_paths:
        # load the image
        image_pil = load_img(image_path, interpolation='nearest', target_size=(image_dim, image_dim, 3))

        # turn it into an array
        image_arr = img_to_array(image_pil)

        # add the image_arr to the images_arr array
        images_arr.append(image_arr)

    # turn it into a numpy arrays so that it can be feed into the model as a batch
    images = np.array(images_arr)

    # make a predictions on the batch
    predictions = model.predict(images, batch_size=batch_size)

    return predictions


def predictions_accuracy(class_keys, label, predictions):
    """
    Determine the accuracy of a set of a image predictions

    class_keys: list of class keys
    label: true class of the predictions
    predictions: array of image predictions
    return: average correct image predictions
    """

    # number of correct predictions
    correct_predictions = 0

    # number of predictions made
    n_predictions = len(predictions)

    # check how many predictions were correct
    for prediction in predictions:
        # determine the most likely class from the prediction
        most_likely_class = np.argmax(prediction)

        # get the label of the prediction
        prediction_label = class_keys[most_likely_class]

        # check if it matches the label
        # if so, increment the counter
        if prediction_label == label:
            correct_predictions += 1

    # calculate the average correct of the predictions
    average = correct_predictions / n_predictions

    return average


def plot_prediction(class_keys, image_paths, predictions):
    """
    Plots image predictions with the most likely class, and the probabilities of the prediction.

    class_keys: list of class keys
    image_paths: path to an image
    predictions: predictions of the image_paths
    """

    for index, image_path in enumerate(image_paths):
        # determine the most likely class from the prediction
        most_likely_class = np.argmax(predictions[index])

        # add class labels for the prediction
        # remember that we feed in a batch so we need to grab the first prediction
        prediction_classes = [str(class_keys[prob_index]) + ": " + str(round(prob * 100, 4)) + "%" for prob_index, prob
                              in enumerate(predictions[index])]

        # generate the prediction label
        subplot_label = "Prediction: " + str(class_keys[most_likely_class]) + "\nProbabilities: " + ', '.join(
            prediction_classes)

        # setup a plot
        fig = plt.figure(figsize=(7, 7), tight_layout=True)
        fig.set_facecolor('white')

        # load the image
        image_pil = load_img(image_path, interpolation='nearest', target_size=(200, 200))

        # render an image to the plot
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image_pil)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(subplot_label)

train_dir = "venv/dataset/train"
validate_dir = "venv/dataset/validate"
# number of images in the training dataset
n_train = 8000

# number of images in the validation dataset
n_validation = 2000

# the number of pixels for the width and height of the image
image_dim = 200

# the size of the image (h,w,c)
input_shape = (image_dim, image_dim, 3)

# the rate which the model learns
learning_rate = 0.001

# size of each mini-batch
batch_size = 32

# nunmber of training episodes
epochs = 10

# directory which we will save training outputs to
# add a timestamp so that tensorboard show each training session as a different run
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
output_logs_dir = 'logs\\' + timestamp + '-' + str(batch_size) + '-' + str(epochs)
print(output_logs_dir)


# directory to save the model
model_name = 'trained_model'

# define data generators
train_data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

validation_data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# tell the data generators to use data from the train and validation directories
train_generator = train_data_generator.flow_from_directory(train_dir,
                                                          target_size=(image_dim, image_dim),
                                                          batch_size=batch_size,
                                                          class_mode='categorical')

validation_generator = validation_data_generator.flow_from_directory(validate_dir,
                                                          target_size=(image_dim, image_dim),
                                                          batch_size=batch_size,
                                                          class_mode='categorical')

# get a dictionary of class names
classes_dictionary = train_generator.class_indices

# turn classes dictionary into a list
class_keys = list(classes_dictionary.keys())

# get the number of classes
n_classes = len(class_keys)

# Get the name of each directory in the root directory and store them as an array.
classes = get_class_labels(validate_dir)

# Get the paths of all the images in the first class directory and store them as a 2d array.
image_paths = get_class_images(classes, validate_dir)

# define the model
# takes in images, convoles them, flattens them, classifies them
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(lr=learning_rate, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# label of the class we are making predictions on
# single_class = class_keys[0]
#
# # first class image paths
# single_class_image_paths = image_paths[0]
#
# # make predictions on the first class
# single_class_predictions = predict(int(n_validation / n_classes), single_class_image_paths, model)
#
# # get the accuracy of predictions on the first class
# single_class_accuracy = predictions_accuracy(class_keys, single_class, single_class_predictions)

for i in range(10):
    single_class_predictions = predict(int(n_train / n_classes), image_paths[i], model)
    single_class_accuracy = predictions_accuracy(class_keys, class_keys[i], single_class_predictions)
    print("Current accuracy of model for class " + class_keys[i] + ": " + str(single_class_accuracy))

# print("Current accuracy of model for class " + single_class + ": " + str(single_class_accuracy))

# # log information for use with tensorboard
# tensorboard = TensorBoard(log_dir=output_logs_dir)

model.fit_generator(train_generator,
                    steps_per_epoch=math.floor(n_train/batch_size),
                    validation_data=validation_generator,
                    validation_steps=n_validation,
                    epochs=epochs,)

for i in range(10):
    single_class_predictions = predict(int(n_train / n_classes), image_paths[i], model)
    single_class_accuracy = predictions_accuracy(class_keys, class_keys[i], single_class_predictions)
    print("Current accuracy of model for class " + class_keys[i] + ": " + str(single_class_accuracy))

# # make predictions on the first class
# single_class_predictions = predict(int(n_train / n_classes), single_class_image_paths, model)
#
# # get the accuracy of predictions on the first class
# single_class_accuracy = predictions_accuracy(class_keys, single_class, single_class_predictions)
#
# print("Current accuracy of model for class " + single_class + ": " + str(single_class_accuracy))

# get 1 image path per class
predict_image_paths = [image_path[0] for image_path in image_paths]

# Make 1 prediction per class
predictions = predict(10, predict_image_paths, model)

# plot the image that was predicted
plot_prediction(class_keys, predict_image_paths, predictions)

model.save('trained_model.h5')
json_string = model.to_json()

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

