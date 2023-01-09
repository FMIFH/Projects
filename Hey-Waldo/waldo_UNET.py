import random

import tensorflow as tf

from keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras import backend as K

IMAGES_PATH = "original-images\\data"
LABEL_PATH = "original-images\\label"

image_height = 1024
image_width = 1024
sub_images_height = 1024
sub_images_width = 1024

images_ids = [os.path.join(IMAGES_PATH,i) for i in next(os.walk(IMAGES_PATH))[2]]
label_ids = [os.path.join(LABEL_PATH,i) for i in next(os.walk(LABEL_PATH))[2]]

images = np.array([np.array(Image.open(img).convert('L').resize((image_height, image_width))) for img in images_ids], dtype=np.float32)
labels = np.array([np.array(Image.open(img).resize((image_height, image_width))) for img in label_ids], dtype=np.float32)

sub_images = [i[x:x+sub_images_width,y:y+sub_images_height] for i in images for x in range(0, image_width, sub_images_width) for y in range(0, image_height, sub_images_height)]
sub_labels = [i[x:x+sub_images_width,y:y+sub_images_height] for i in labels for x in range(0, image_width, sub_images_width) for y in range(0, image_height, sub_images_height)]

sub_images = np.array(sub_images)
sub_labels = np.array(sub_labels)

shuffle_img, shuffle_lab = shuffle(sub_images, sub_labels)

def unet():
    #Build the model
    inputs = tf.keras.layers.Input((sub_images_height, sub_images_width, 1))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model


def get_image(image):
    diff = int(image_height / sub_images_height)
    img = np.empty((image_height, image_width))
    for i in range(image.shape[0]):
        for x in range(sub_images_width):
            for y in range(sub_images_height):
                img[(i // diff) * sub_images_width + x, (i % diff) * sub_images_height + y] = image[i, x, y] * 255
    img = Image.fromarray(img)
    img.show()

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
  union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2])
  union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

model = unet()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10**-5),
              loss='binary_crossentropy',
              metrics=['accuracy',dice_coef])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('hey_waldo.h5', verbose=1, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.5,patience=2)
]

results = model.fit(shuffle_img, shuffle_lab, validation_split=0.1, batch_size=2, epochs=100, callbacks=callbacks)
test = 2

total_panel = int((image_height*image_width)/(sub_images_height*sub_images_width))
pred = model.predict(sub_images[total_panel*test:total_panel*(test+1)])
true = sub_labels[total_panel*test:total_panel*(test+1)]
og = sub_images[total_panel*test:total_panel*(test+1)]

diff = int(image_height / sub_images_height)
img = np.empty((image_height, image_width))
for i in range(pred.shape[0]):
    for x in range(pred.shape[1]):
        for y in range(pred.shape[2]):
            img[(i // diff)*sub_images_width+x, (i % diff)*sub_images_height+y] = 0 if pred[i, x, y] < 0.5 else 255
img = Image.fromarray(img)
img.show()

get_image(true)
img = Image.open(images_ids[test])
img.show()

