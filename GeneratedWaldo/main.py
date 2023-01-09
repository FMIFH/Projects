import random
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import os

from matplotlib import pyplot as plt
from keras.utils import Sequence

BACKGROUND = 'background'
WALDO = 'waldo'
WALDO_DIM = (40,40)
BACKGROUND_DIM = (533, 300)
COLOR_MODE = 'RGB'
COLOR_CHANNELS = len(COLOR_MODE)


class DataGen(Sequence):
    def __init__(self, batch_size=16,steps=100):
        self.batch_size = batch_size
        self.steps = steps
        self.epoch = 0

    def __len__(self):
        return int(self.steps)

    def on_epoch_end(self):
        self.epoch = self.epoch+1
        if self.epoch % 5 == 0:
            self.test_model(self.epoch)

    def __getitem__(self, index):
        x_batch = np.zeros((self.batch_size, BACKGROUND_DIM[1], BACKGROUND_DIM[0], COLOR_CHANNELS))
        boundary_box = np.zeros((self.batch_size, 2))

        for i in range(self.batch_size):
            # generate an example image
            sample_im, pos = self.generate_sample_image()
            #if index == 0: self.plot_bounding_box(sample_im.convert('L'), pos).show()
            # put the images to the arrays
            x_batch[i] = np.array(sample_im.convert(COLOR_MODE), dtype=np.uint8)
            boundary_box[i, 0] = pos[0]
            boundary_box[i, 1] = pos[1]

        return x_batch, boundary_box

    def generate_sample_image(self):
        waldo_ids = [os.path.join(WALDO, i) for i in next(os.walk(WALDO))[2]]
        background_ids = [os.path.join(BACKGROUND, i) for i in next(os.walk(BACKGROUND))[2]]

        background = random.choice(background_ids)
        background_im = Image.open(background).resize(BACKGROUND_DIM)

        waldo = random.choice(waldo_ids)
        waldo_im = Image.open(waldo)
        waldo_im = waldo_im.resize(WALDO_DIM)

        col = np.random.randint(0, BACKGROUND_DIM[0]-WALDO_DIM[0])
        row = np.random.randint(0, BACKGROUND_DIM[1]-WALDO_DIM[1])

        background_im.paste(waldo_im, (col, row), mask=waldo_im)
        return background_im, (col, row)

    def plot_bounding_box(self, image, gt_coords, pred_coords=None):
        # convert image to array
        draw = ImageDraw.Draw(image)
        draw.rectangle((gt_coords[0], gt_coords[1],
                        gt_coords[0] + WALDO_DIM[0], gt_coords[1] + WALDO_DIM[1]),
                       outline='green',
                       width=2)

        if pred_coords:
            draw.rectangle((pred_coords[0], pred_coords[1],
                            pred_coords[0] + WALDO_DIM[0], pred_coords[1] + WALDO_DIM[1]),
                           outline='red',
                           width=2)
        return image

    def test_model(self, epoch):
        for i in range(3):
            # get sample image
            sample_im, pos = self.generate_sample_image()
            sample_image_normalized = np.array(sample_im.convert(COLOR_MODE), dtype=np.uint8).reshape(1, BACKGROUND_DIM[1], BACKGROUND_DIM[0], COLOR_CHANNELS)
            predicted_box = model.predict(sample_image_normalized)

            # get bounding boxes
            im = self.plot_bounding_box(sample_im, pos, (predicted_box[0][0], predicted_box[0][1]))
            im.show()
            im.save("{epoch}-{iteration}.jpg".format(epoch=epoch,iteration=i))


def convolutional_block(inputs):
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(s)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 6, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 6, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)

    return x


def regression_block(x):
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(2)(x)
    return x

def plot_bounding_box( image, gt_coords, pred_coords=None):
    # convert image to array
    draw = ImageDraw.Draw(image)
    draw.rectangle((gt_coords[0], gt_coords[1],
                    gt_coords[0] + WALDO_DIM[0], gt_coords[1] + WALDO_DIM[1]),
                   outline='green',
                   width=2)

    if pred_coords:
        draw.rectangle((pred_coords[0], pred_coords[1],
                        pred_coords[0] + WALDO_DIM[0], pred_coords[1] + WALDO_DIM[1]),
                       outline='red',
                       width=2)
    return image

inputs = tf.keras.Input((BACKGROUND_DIM[1], BACKGROUND_DIM[0], COLOR_CHANNELS))
x = convolutional_block(inputs)
box_output = regression_block(x)
model = tf.keras.Model(inputs=inputs, outputs=box_output)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_delta=1, min_lr=10**-7)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
callbacks = [reduce_lr, early_stop]

"""
train_generator = DataGen(batch_size=32,steps=50)
hist = model.fit(train_generator,
                 epochs=100,
                 callbacks=callbacks,
                 verbose=1)
model.save('waldo.hdf5')
"""
model.load_weights('waldo.hdf5')
img = Image.open('t1.jpg').resize(BACKGROUND_DIM)
im_array = np.array(img.convert(COLOR_MODE), dtype=np.uint8).reshape(1, BACKGROUND_DIM[1], BACKGROUND_DIM[0], COLOR_CHANNELS)
pred = model.predict(im_array)[0]
print(pred)
plot_bounding_box(img,pred).show()

