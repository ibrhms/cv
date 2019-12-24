import numpy as np
import cv2 
import math

class bg_extractor_1:
    def __init__(self, image_shape, resize_ratio = 8, kernel_size = 9, min_contour_area = 256, show_images = False, find_contour = False):
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
        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel - 4)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
        if self.show_images:
            cv2.imshow("mask", closing)
        return closing

    def find_biggest_area(self, mask):
        im2, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                loc = cv2.boundingRect(contour)
                result.append(loc)

        return result    
    
    def find_moving_objects(self,frame):
        image = cv2.resize(frame, (self.img_wr, self.img_hr))

        mask = self.get_mask(image=image)
        locs = self.find_biggest_area(mask=mask)
        
        detected = False
        for loc in locs:
            loc = [x * self.resize_ratio for x in loc]
            x,y,w,h = loc
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected = True
            
        if self.show_images:
            cv2.imshow('result',frame)
        return detected, locs
        
    def find_moving_objects_2(self,frame):
        image = cv2.resize(frame, (self.img_wr, self.img_hr))

        mask = self.get_mask(image=image)
        locs = self.find_biggest_area(mask=mask)
        
        detected = False
        for loc in locs:

            loc = [x * self.resize_ratio for x in loc]
            x,y,w,h = loc[0],0,loc[2],self.img_h
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected = True
            
        if self.show_images:
            cv2.imshow('result',frame)
        return detected, locs
        
if __name__ == "__main__":
    pass
 