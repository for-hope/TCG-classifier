import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

import cv2
import os
import time

import numpy as np

print("Starting classifier...")

execution_path = os.getcwd()
labels = ['ygo', 'mtg']
img_width = 223
img_height = 310


start = "save"
dir = 'test_images\mtg'
model_path = 'models\model-5-1624194872.8838975'

if start == "load":
    
    model = keras.models.load_model(model_path)
    # ? switch this to MTG dir or YGO idr
    
    for img in os.listdir(dir):
        img = os.path.join(dir, img)
        # convert BGR to RGB format
        img_arr = cv2.imread(img)
        # Reshaping images to preferred size
        resized_arr = cv2.resize(img_arr, (img_width, img_height))
        x_val = [resized_arr]
        x_val = np.array(x_val) / 255
        x_val.reshape(-1, img_width, img_height, 1)
        prediction = model.predict_classes(x_val)
        print("################################")
        print(f"The result for {img} is {labels[prediction[0]]}")
        print("################################")
    exit()


def get_data(data_dir):
    print("Getting data...")
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                # convert BGR to RGB format
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                # Reshaping images to preferred size
                resized_arr = cv2.resize(img_arr, (img_width, img_height))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    print("Done getting data")
    return np.array(data, dtype=np.object)


train = get_data(os.path.join(execution_path, 'data', 'train'))
val = get_data(os.path.join(execution_path, 'data', 'test'))

l = []
for i in train:
    if(i[1] == 0):
        l.append("mtg")
    else:
        l.append("ygo")
print("L length is : " + str(len(l)))

sns.set_style('darkgrid')
sns.countplot(l)


plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])

plt.show()

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_width, img_height, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_width, img_height, 1)
y_val = np.array(y_val)


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=30,
    zoom_range=0.2,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


# * Training the model:

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu",
          input_shape=(img_height, img_width, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = Adam(lr=0.000001)
n_epochs = 5
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=n_epochs,
                    validation_data=(x_val, y_val))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(n_epochs)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# ! `model.predict_classes()` is deprecated
#predictions = model.predict_classes(x_val)
predictions = np.argmax(model.predict(x_val), axis=-1)
predictions = predictions.reshape(1, -1)[0]
print(classification_report(y_val, predictions,
      target_names=['MTG (Class 0)', 'YGO (Class 1)']))

model.save('models/model-' + str(n_epochs) + '-' + str(time.time()))
