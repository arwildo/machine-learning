#!/bin/python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


data = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

for train in range(len(x_train)):
    for row in range(28):
        for x in range(28):
            if x_train[train][row][x] != 0:
                x_train[train][row][x] = 1


model = tf.keras.models.load_model('char_guesser.model')
print(len(x_test))
predictions = model.predict(x_test[:10])

count = 0
for x in range(len(predictions)):
    guess = (np.argmax(predictions[x]))
    actual = y_test[x]
    print("Prediction letter = ", guess)
    print("Actual letter = ", actual)
    
    if guess != actual:
        count += 1

    plt.imshow(x_test[x], cmap=plt.cm.binary)
    plt.show()

print("Wrong guesset = ", count, "from ", len(x_test), "test")
print(str(100 - (count/len(x_test)*100) + '% correct'))
