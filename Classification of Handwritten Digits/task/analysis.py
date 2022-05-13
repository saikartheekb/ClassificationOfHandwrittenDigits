import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train = np.array(x_train).reshape(60000, 784)

print(f"Classes: {np.unique(y_train)}")
print(f"Features' shape: {x_train.shape}")
print(f"Target's shape: {y_train.shape}")
print(f"min: {x_train.min()}, max: {x_train.max()}")