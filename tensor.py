import tensorflow as tf
print(tf.__version__)

# se guarda el dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# se cargan los datos de entrenamiento y prueba
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200)


# Set index of image to be seen
img_index = 5

# Plot image
plt.imshow(training_images[img_index], cmap='gray')
plt.axis(False)

print("Label:", training_labels[img_index])
print("Matrix:", training_images[img_index])


training_images  = training_images / 255.0
test_images = test_images / 255.0

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=200)


# Set index of image to be seen
img_index = 5

# Plot image
plt.imshow(training_images[img_index], cmap='gray')
plt.axis(False)

print("Label:", training_labels[img_index])
print("Matrix:", training_images[img_index])


training_images[0].shape

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(256, activation=tf.nn.relu), 
    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
    tf.keras.layers.Dense(64, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(training_images, training_labels, epochs=8)

model.evaluate(test_images, test_labels)

import random

test_index = random.randint(0, 10000-1)

plt.imshow(test_images[test_index], cmap='Spectral')
plt.axis(False)
input_image = np.reshape(test_images[test_index], (1,784))
print("Label:", test_labels[test_index])
prediction = model.predict(np.expand_dims(input_image, axis=-1))
print("Prediction:", np.argmax(prediction))

prediction

