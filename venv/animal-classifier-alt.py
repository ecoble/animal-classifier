import tensorflow as tf
import skimage
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random

import keras
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import optimizers

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".jpeg")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(str(d))
    return images, labels


ROOT_PATH = "./dataset/"
train_data_directory = os.path.join(ROOT_PATH, "train")
test_data_directory = os.path.join(ROOT_PATH, "validate")

images, labels = load_data(train_data_directory)
test_images, test_labels = load_data(test_data_directory)

images_array = np.array(images)
labels_array = np.array(labels)
test_images_array = np.array(test_images)
test_labels_array = np.array(test_labels)

images_array = images_array / 255.0
test_images_array = test_images_array / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
    plt.xlabel(labels[i])
plt.show()



# images28 = [transform.resize(image, (200, 200)) for image in images]
#
# # Convert `images28` to an array
# images28 = np.array(images28)
#
# # Convert `images28` to grayscale
# images28 = rgb2gray(images28)



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(7266, 200, 200, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(images_array, labels_array, epochs=10)

test_loss, test_acc = model.evaluate(test_images_array, test_labels_array)

print('\nTest accuracy:', test_acc)

# x = tf.placeholder(dtype = tf.float32, shape = [None, 200, 200])
# y = tf.placeholder(dtype = tf.int32, shape = [None])
#
# # Flatten the input data
# images_flat = tf.contrib.layers.flatten(x)
#
# # Fully connected layer
# logits = tf.contrib.layers.fully_connected(images_flat, 10, tf.nn.relu)
#
# # Define a loss function
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
#                                                                     logits = logits))
# # Define an optimizer
# train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#
# # Convert logits to label indexes
# correct_pred = tf.argmax(logits, 1)
#
# # Define an accuracy metric
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# tf.set_random_seed(1234)
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())
#
# for i in range(201):
#         print('EPOCH', i)
#         _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
#         if i % 10 == 0:
#             print("Loss: ", loss)
#         print('DONE WITH EPOCH')




# Get the unique labels
# images28 = [transform.resize(image, (200, 200)) for image in images]
# images28 = np.array(images28)
#
# print("Here2")
#
# # Initialize placeholders
# x = tf.placeholder(dtype = tf.float32, shape = [None, 200, 200, 3])
# y = tf.placeholder(dtype = tf.int32, shape = [None])
#
# # Flatten the input data
# images_flat = tf.contrib.layers.flatten(x)
#
# # Fully connected layer
# logits = tf.contrib.layers.fully_connected(images_flat, 10, tf.nn.relu)
#
# # Define a loss function
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
#                                                                     logits = logits))
# # Define an optimizer
# train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#
# # Convert logits to label indexes
# correct_pred = tf.argmax(logits, 1)
#
# # Define an accuracy metric
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# tf.set_random_seed(1234)
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())
#
# print("Here3")
# for i in range(201):
#         print('EPOCH', i)
#         _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
#         if i % 10 == 0:
#             print("Loss: ", loss)
#         print('DONE WITH EPOCH')
#
# sample_indexes = random.sample(range(len(images28)), 10)
# sample_images = [images28[i] for i in sample_indexes]
# sample_labels = [labels[i] for i in sample_indexes]
#
# # Run the "correct_pred" operation
# predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
#
# # Print the real and predicted labels
# print(sample_labels)
# print(predicted)
#
# test_images, test_labels = load_data(test_data_directory)
#
# # Transform the images to 28 by 28 pixels
# test_images28 = [transform.resize(image, (200, 200)) for image in test_images]
#
# # Convert to grayscale
# # test_images28 = rgb2gray(np.array(test_images28))
#
# # Run predictions against the full test set.
# predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]
#
# # Calculate correct matches
# match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
#
# # Calculate the accuracy
# accuracy = match_count / len(test_labels)
#
# # Print the accuracy
# print("Accuracy: {:.3f}".format(accuracy))