import os
import random
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Lambda, Input, BatchNormalization

COLOR_FORMAT = 'L'
IMAGE_WIDTH = IMAGE_HEIGHT = 256
IMAGE_CHANNELS = len(COLOR_FORMAT)

WALDO_IMG = "{dim}\\waldo".format(dim=IMAGE_HEIGHT)
NOT_WALDO_IMG = "{dim}\\notwaldo".format(dim=IMAGE_HEIGHT)

TEST_IMAGE = 'original-images\\data\\7.jpg'

waldo_ids = [os.path.join(WALDO_IMG,i) for i in next(os.walk(WALDO_IMG))[2]]
not_waldo_ids = [os.path.join(NOT_WALDO_IMG,i) for i in next(os.walk(NOT_WALDO_IMG))[2]]

waldo_img = np.array([np.array(Image.open(img).convert(COLOR_FORMAT)) for img in waldo_ids], dtype=np.int8)
waldo_label = np.ones(waldo_img.shape[0])
not_waldo_img = np.array([np.array(Image.open(img).convert(COLOR_FORMAT)) for img in not_waldo_ids], dtype=np.int8)
not_waldo_label = np.zeros(not_waldo_img.shape[0])

X_data = np.concatenate((waldo_img,not_waldo_img),axis=0)
y_data = np.concatenate((waldo_label,not_waldo_label),axis=0)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1)


def get_model():
    model = Sequential()
    model.add(Input((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
    model.add(Lambda(lambda x: x / 255))

    model.add(Conv2D(128, (2, 2), activation=tf.keras.layers.LeakyReLU()))
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (2, 2), 1, activation=tf.keras.layers.LeakyReLU()))
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (2, 2), 1, activation=tf.keras.layers.LeakyReLU()))
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (2, 2), 1, activation=tf.keras.layers.LeakyReLU()))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation=tf.keras.layers.LeakyReLU()))
    model.add(Dense(1, activation=tf.keras.activations.sigmoid))
    return model


model = get_model()
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10**-6),
              loss='binary_crossentropy',
              metrics=['accuracy'])
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('hey_waldo2.h5', verbose=1, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2)
]

history = model.fit(X_train, y_train, validation_split=0.2, epochs=1000, callbacks=callbacks)
res = model.evaluate(X_test, y_test, verbose=1)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy')
plt.legend(loc='lower right')
plt.show()

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

test_im = Image.open(TEST_IMAGE)

newW = test_im.size[0] + ((IMAGE_WIDTH - test_im.size[0] % IMAGE_WIDTH) if test_im.size[0] % IMAGE_WIDTH > 0 else 0)
newH = test_im.size[1] + ((IMAGE_HEIGHT - test_im.size[1] % IMAGE_HEIGHT) if test_im.size[1] % IMAGE_HEIGHT > 0 else 0)

reshaped_img = resize_with_padding(test_im,(newW,newH)).convert(COLOR_FORMAT)
test_image_array = np.array(reshaped_img)
sub_images = [test_image_array[x:x+IMAGE_WIDTH,y:y+IMAGE_HEIGHT]
              for x in range(0, test_image_array.shape[0], IMAGE_WIDTH)
              for y in range(0, test_image_array.shape[1], IMAGE_HEIGHT)]
sub_images = np.array(sub_images)


pred = model.predict(waldo_img)
print(pred)
waldo_tile = [i for i in range(pred.shape[0]) if pred[i,0] > 0.5]
print(waldo_tile)
for t in waldo_tile:
    Image.fromarray(waldo_img[t],mode=COLOR_FORMAT).show()
