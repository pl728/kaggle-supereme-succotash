import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.utils.np_utils import to_categorical

# csv -> tensors (categorical becomes integers, numerical become floating point)
# -> normalized to mean 0, var 1

# get X_train, y_train
X_train = pd.read_csv("datasets/train.csv")
test = np.array(pd.read_csv("datasets/test.csv"))
y_train = X_train.pop('label')
X_train = np.array(X_train)

# normalization
X_train = X_train / 255.0
test = test / 255.0

# label encoding
y_train = to_categorical(y_train, num_classes=10)

# reshape image to three dimensions
X_train = X_train.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)

# get training and validation set using sklearn
random_seed = 2
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)

# building the model
# TODO: review
# conv2d -> conv2d -> maxpool2d -> dropout ->
# conv2d -> conv2d -> maxpool2d -> dropout -> flatten -> dense -> dropout -> dense

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# optimizer and annealer
model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
              loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=128, epochs=1, validation_data=(X_val, y_val   ), verbose=2)
results = model.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

submission.to_csv("submission.csv", index=False)
