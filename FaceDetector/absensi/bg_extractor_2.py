import numpy as np
import cv2 
import math

class bg_extractor_2:
    def __init__(self, image_shape, resize_ratio = 8, kernel_size = 3, min_contour_area = 256, show_images = False, find_contour = False):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        self.show_images = show_images

        self.find_contour = find_contour

        self.resize_ratio = resize_ratio
        self.img_h, self.img_w, self.img_c = image_shape
        self.img_hr = int(self.img_h / resize_ratio)
        self.img_wr = int(self.img_w / resize_ratio)
        self.img_ar = self.img_hr * self.img_wr
        
        self.min_moving_area = int(math.ceil((self.img_ar/min_contour_area)))
    
    def get_mask(self, image):
        fg_mask = self.fgbg.apply(image)
        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
        if self.show_images:
            cv2.imshow("mask", closing)
        return closing

    def find_moving_objects(self,image):
        image = cv2.resize(image, (self.img_wr, self.img_hr))

        mask = self.get_mask(image=image)

        num_of_nonzero = cv2.countNonZero(mask)

        detected = False
        if num_of_nonzero >=self.min_moving_area:
            if self.show_images:
                cv2.imshow('result',image)
            detected = True
        return detected, image