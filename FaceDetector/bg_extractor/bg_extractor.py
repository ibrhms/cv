import numpy as np
import cv2 
import math

class bg_extractor_1:
    def __init__(self, image_shape, resize_ratio = 8, kernel_size = 7, min_contour_area = 256, show_images = False, find_contour = False):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        self.show_images = show_images

        self.find_contour = find_contour

        self.resize_ratio = resize_ratio
        self.img_h, self.img_w, self.img_c = image_shape
        self.img_hr = int(self.img_h / resize_ratio)
        self.img_wr = int(self.img_w / resize_ratio)
        self.img_ar = self.img_hr * self.img_wr
        
        self.min_contour_area = int(math.ceil((self.img_ar/min_contour_area)))
    
    def get_mask(self, image):
        fg_mask = self.fgbg.apply(image)
        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel - 2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
        if self.show_images:
            cv2.imshow("mask", closing)
        return closing

    def find_biggest_area(self, mask):
        im2, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                x,y,w,h = cv2.boundingRect(contour)
                loc = [x,y,w,h]
                result.append(loc)

        if self.show_images:
            cv2.drawContours(mask, contours, -1, (0,255,0), 3)
            cv2.imshow("contours", mask)
        return result    
    
    def find_moving_objects(self,image):
        image = cv2.resize(image, (self.img_wr, self.img_hr))

        mask = self.get_mask(image=image)
        locs = self.find_biggest_area(mask=mask)
        
        detected = False
        for loc in locs:
            x,y,w,h = loc
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected = True
            
        if self.show_images:
            cv2.imshow('result',image)
        return detected, image

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