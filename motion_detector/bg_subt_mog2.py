import numpy as np
import cv2 as cv

class bg_sub_mog2:
    def __init__(self, op_rad = 7, cl_rad = 7):
        self.bg_sub = cv.bgsegm.createBackgroundSubtractorMOG()
        self.op_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(op_rad,op_rad))
        self.cl_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(cl_rad,cl_rad))
    
    def process_mask(self, frame):
        fgmask = self.bg_sub.apply(frame)
        #_,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
        opening = cv.morphologyEx(fgmask, cv.MORPH_OPEN, self.op_kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, self.cl_kernel)
        return closing

    def isObjectsPresent(self, frame, thr = 0.1):
        mask = self.process_mask(frame)
        num_of_nonzero = np.count_nonzero(mask)
        ratio = num_of_nonzero/mask.size
        if ratio < thr:
            return False, mask
        else:
            return True, mask

