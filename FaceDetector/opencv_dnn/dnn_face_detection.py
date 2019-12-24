import numpy as np 
import cv2 
from datetime import datetime
from cv_utilities import rectangler
import os
import math

def slicing_window(image, step_size, window_size, padding=True):
    if padding:
        num_window_y = math.ceil(image.shape[0] / step_size)
        num_window_x = math.ceil(image.shape[1] / step_size)

        num_padding_y = num_window_y * step_size - image.shape[0]
        num_padding_x = num_window_x * step_size - image.shape[1]
        
        top_pad = int(math.floor(num_padding_x * 0.5))
        bottom_pad = num_padding_x - top_pad

        left_pad = int(math.floor(num_padding_y * 0.5))
        right_pad = num_padding_y - left_pad

        image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad,cv2.BORDER_CONSTANT, None, [0,0,0])

    return_images = []
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            target = image[y:y+step_size, x:x+step_size]
            return_images.append(target)

    return return_images

class opencv_dnn_face_det:
    def __init__(self, prototxt_filename, weights):
        self.net = cv2.dnn.readNetFromCaffe(prototxt_filename, weights)

    def detect_faces(self, image):
        images = slicing_window(image, 300,300)
        for frame in images:
            blob = cv2.dnn.blobFromImage(
                image = cv2.resize(image, (300, 300)), 
                scalefactor = 1.0,
                size = (300, 300), 
                mean = (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            return detections

if __name__ == "__main__":
    a = cv2.imread("badoit_rouge.jpg")
    a_s = slicing_window(image = a.copy(), step_size = 100, window_size =(300,300))
    for idx, image in enumerate(a_s):
        cv2.imwrite(str(idx)+".jpg", image)