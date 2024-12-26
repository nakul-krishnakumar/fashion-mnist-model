import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data

# 60k imgs are used to train the model,
# 10k imgs are used to test the trained model
# no cross validation set
(train_imgs, train_labels), (test_imgs, test_labels) = load_data()

"""
    --------------------- LABELS ---------------------
    | 0  ->  T-shirt/top                              |
    | 1  ->  Trouser                                  |
    | 2  ->  Pullover                                 |
    | 3  ->  Dress                                    |
    | 4  ->  Coat                                     |
    | 5  ->  Sandal                                   |
    | 6  ->  Shirt                                    |
    | 7  ->  Sneaker                                  |
    | 8  ->  Bag                                      |
    | 9  ->  Ankle boot                               |
    --------------------------------------------------
"""

print(f'Shape of image training set: {train_imgs.shape}')
print(f'Shape of label training set: {train_labels.shape}')
print(f'Shape of image testing set: {test_imgs.shape}')
print(f'Shape of label testing set: {test_labels.shape}')