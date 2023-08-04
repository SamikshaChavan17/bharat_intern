#!/usr/bin/env python
# coding: utf-8

# In[ ]:


BHARAT INTERN 
SAMIKSHA CHAVAN 
K.K.WAGH COLLEGE 


# In[3]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# In[ ]:


'''The output of the code provided will include the training progress and the final test accuracy of the trained model. Here's what the output might look like:
'''
Epoch 1/10
750/750 [==============================] - 2s 2ms/step - loss: 0.3366 - accuracy: 0.9021 - val_loss: 0.1713 - val_accuracy: 0.9513
Epoch 2/10
750/750 [==============================] - 1s 1ms/step - loss: 0.1447 - accuracy: 0.9579 - val_loss: 0.1312 - val_accuracy: 0.9613
...
Epoch 10/10
750/750 [==============================] - 1s 1ms/step - loss: 0.0352 - accuracy: 0.9887 - val_loss: 0.0982 - val_accuracy: 0.9712
313/313 [==============================] - 0s 725us/step - loss: 0.0820 - accuracy: 0.9756
Test accuracy: 0.975600004196167
```
'''
In this example output, you can see the training progress for each epoch, including the training loss, 
training accuracy, validation loss, and validation accuracy. After training is complete, the model is
evaluated on the test data, and the final test accuracy is printed, which in this case is approximately 97.56%.
Keep in mind that the exact numbers might vary due to random initialization and other factors,
but the output structure will be similar.'''

