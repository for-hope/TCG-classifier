from scanner import detect_image
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
from PIL import Image
import scanner
import numpy as np
import imutils
print("Starting classifier...")


execution_path = os.getcwd()
labels = ['ygo', 'mtg', 'poki']
img_width = 223
img_height = 310
cropped_size = 6

start = "load"
# dir = 'test_images\mtg'
model_path = 'models\model-5-1625169312.1249607'

if start == "load":

    model = keras.models.load_model(model_path)
    # ? switch this to MTG dir or YGO idr
    dir = 'test_images\mtg'
    for img in os.listdir(dir):
        img = os.path.join(dir, img)
        # convert BGR to RGB format
        img_arr = cv2.imread(img)
        full_image = img_arr
        try:
            img_arr, full_image = detect_image(img)
            
        except:
            print("Error")

        #cv2.imshow("Scanned", imutils.resize(img_arr, height = 650))
        # Reshaping images to preferred size
        resized_arr = cv2.resize(img_arr, (img_width, img_height))
        #cv2.putText(frame, class_names[cls], (xy[0], xy[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1) 


        #! change this to the cropped images and need to detect images as well
        x_val = [resized_arr]
        x_val = np.array(x_val) / 255
        x_val.reshape(-1, img_width, img_height, 1)
        prediction = model.predict_classes(x_val)
        print("################################")
        print(f"The result for {img} is {labels[prediction[0]]}")
        print("################################")
        resized_arr = cv2.putText(full_image, labels[prediction[0]], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.imshow("Detected", resized_arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
                resized_arr = cv2.resize(img_arr, (img_width, img_height))
                image = Image.fromarray(resized_arr)
                box = (cropped_size, cropped_size, img_width -
                       cropped_size, img_height-cropped_size)
                cropped_image = image.crop(box)
                # Reshaping images to preferred size

                cropped_image = np.array(cropped_image)
                # x = 30
                # y = 30
                resized_arr = cv2.resize(
                    cropped_image, (img_width, img_height))
                #!  img[int(y):int(y+h), int(x):int(x+w)]
                #resized_arr = resized_arr[int(y):int(y+img_height), int(x):int(x+img_width)]
                # plt.show()
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
    elif i[1] == 1:
        l.append("ygo")
    else:
        l.append("poki")

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


#! need more intense image augmentation
datagen = ImageDataGenerator(
    brightness_range=[0.8,1.0],
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=50,
    zoom_range=0.3, #0.2  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.4,
    
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.4,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images


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
model.add(Dense(3, activation="softmax"))

model.summary()

opt = Adam(lr=0.00001)
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
      target_names=['MTG (Class 0)', 'YGO (Class 1)', 'Poki (Class 2)']))


model.save('models/model-' + str(n_epochs) + '-' + str(time.time()))
print("Saved model as " + str(n_epochs) + '-' + str(time.time()))
