import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data

# Setting up callbacks to stop training if accuracy crosses 80%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.8):
            print("\nReached 80% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

# 60k imgs are used to train the model,
# 10k imgs are used to test the trained model
# no cross validation set
(train_imgs, train_labels), (test_imgs, test_labels) = load_data()

labels = [
    "t-shirt/top", #0
    "Trouser",     #1
    "Pullover",    #2
    "Dress",       #3
    "Coat",        #4
    "Sandal",      #5
    "Shirt",       #6
    "Sneaker",     #7
    "Bag",         #8
    "Ankle boot"   #9
]

print(f'Shape of image training set: {train_imgs.shape}')
print(f'Shape of label training set: {train_labels.shape}')
print(f'Shape of image testing set: {test_imgs.shape}')
print(f'Shape of label testing set: {test_labels.shape}')

# Normalize the training & testing image set so model works better
train_imgs = train_imgs / 255.0
test_imgs = test_imgs / 255.0

# Building the model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

# Configuring the model
# Choosing the optimizer and loss function
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Input the training data and mention the number of epochs
model.fit( train_imgs, train_labels, epochs=5, callbacks=[callbacks] )

# Test the model on the testing data
print("\n------- EVALUATION ON TEST SET ---------------\n")
model.evaluate(test_imgs, test_labels)

a = int(input("\nEnter test label index to predict: "))

# Making the prediction
predicted_label = model.predict(np.expand_dims(test_imgs[a], axis=0))

 
index = np.argmax(predicted_label)
print(f"Predicted Label: {labels[index]}")
print(f"Actual Label: {labels[test_labels[a]]}")
