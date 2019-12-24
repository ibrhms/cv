import numpy as np 
import math
import cv2

def rectangler(x_start, y_start, x_end, y_end, img_width, img_heights):
    
    width = x_end - x_start 
    mid_x = x_start + math.floor(width *0.5)
    height = y_end - y_start
    mid_y = y_start + math.floor(height *0.5)

    if (height > width):
        new_width = height
        half_new_width = math.floor(new_width * 0.5)
        mod_x_start = mid_x-half_new_width
        while (mod_x_start < 0):
            half_new_width -= 1
            mod_x_start = mid_x - half_new_width

        new_width = half_new_width *2
        new_height = new_width
        new_x_start = mod_x_start
        new_y_start = mid_y - half_new_width
    else:
        new_height = width
        half_new_height = math.floor(new_height * 0.5)
        mod_y_start = mid_y - half_new_height
        while mod_y_start<0:
            half_new_height -= 1
            mod_y_start = mid_y - half_new_height

        new_height = half_new_height * 2
        new_width = new_height
        new_y_start = mod_y_start
        new_x_start = mid_x - half_new_height

    return new_x_start, new_y_start, new_x_start + new_width, new_y_start + new_height

