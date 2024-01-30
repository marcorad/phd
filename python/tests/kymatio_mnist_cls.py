#https://www.kymat.io/gallery_2d/mnist_keras.html#sphx-glr-gallery-2d-mnist-keras-py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

from kymatio.keras import Scattering2D

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255., x_test / 255.

inputs = Input(shape=(28, 28))
x = Scattering2D(J=3, L=8)(inputs)
x = Flatten()(x)
x_out = Dense(10, activation='softmax')(x)
model = Model(inputs, x_out)

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train[:40000], y_train[:40000], epochs=100,
          batch_size=128, validation_split=0.2)

model.evaluate(x_test, y_test)