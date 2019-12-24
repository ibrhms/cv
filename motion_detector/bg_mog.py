import numpy as np
import cv2 as cv
import time
from datetime import datetime 
from  bg_subt_mog2 import bg_sub_mog2

ratio_thr = 0.01

cap = cv.VideoCapture(0)

fgbg = bg_sub_mog2(op_rad=3,cl_rad=3)

while(1):
    time.sleep(0.1)
    ret, frame = cap.read()
    isDetected, fgmask = fgbg.isObjectsPresent(frame, ratio_thr)
    if isDetected:
        print("object detected at :",datetime.now())
        cv.imshow('frame',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()