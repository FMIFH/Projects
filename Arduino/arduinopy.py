import numpy as np
import pyfirmata
import os
import time

import cv2
RIGHT_SIDE_GAS = 5
RIGHT_SIDE_REVERSE = 4
LEFT_SIDE_GAS = 2
LEFT_SIDE_REVERSE = 3


def turn_left():
    board.digital[LEFT_SIDE_GAS].write(0)
    board.digital[RIGHT_SIDE_GAS].write(1)
    board.digital[LEFT_SIDE_REVERSE].write(1)
    board.digital[RIGHT_SIDE_REVERSE].write(0)


def turn_right():
    board.digital[LEFT_SIDE_GAS].write(1)
    board.digital[RIGHT_SIDE_GAS].write(0)
    board.digital[LEFT_SIDE_REVERSE].write(0)
    board.digital[RIGHT_SIDE_REVERSE].write(1)


def go_straight():
    board.digital[LEFT_SIDE_GAS].write(1)
    board.digital[RIGHT_SIDE_GAS].write(1)
    board.digital[LEFT_SIDE_REVERSE].write(0)
    board.digital[RIGHT_SIDE_REVERSE].write(0)


def reset():
    board.digital[LEFT_SIDE_GAS].write(0)
    board.digital[RIGHT_SIDE_GAS].write(0)
    board.digital[LEFT_SIDE_REVERSE].write(0)
    board.digital[RIGHT_SIDE_REVERSE].write(0)


board = pyfirmata.Arduino('COM3')
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, cam = vc.read()
else:
    rval = False
"""
reset()

while True:
    t_end = time.time() + 15
    print("TURNING RIGHT")
    while time.time() < t_end:
        turn_right()

    t_end = time.time() + 15
    print("TURNING LEFT")
    while time.time() < t_end:
        turn_left()

reset()
"""
while rval:
    frame = cv2.resize(cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY),(64,64))
    band = frame[32]
    os.system('cls')
    cv2.imshow('Gray image',cam)
    try:
        tape = int(np.mean(np.where(band < min(band)+10)))
    except:
        tape = 0
    if tape < 21:
        print("Left")
        turn_left()
    elif tape > 42:
        print("Right")
        turn_right()
    else:
        print("Straight")
        go_straight()
    rval, cam = vc.read()
    key = cv2.waitKey(10)

    if key == 27:  # exit on ESC
        reset()
        break
