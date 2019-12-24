import cv2 
import numpy as np
import os

haar_face_file = "haarcascade_frontalface_default.xml"

class Haar_Detector:
    def __init__(self, detector_xml, verbose=False):
        self.face_cascade = cv2.CascadeClassifier()
        self.detector_loaded = self.face_cascade.load(detector_xml)
        self.verbose = verbose
        if verbose:
            print("detector available :" + str(self.detector_loaded))

    def find_structure(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        structures = self.face_cascade.detectMultiScale(gray, 1.25, 6)
        return structures
    
    def find_and_draw_structure(self, img):
        structures = self.find_structure(img)
        if len(structures) > 0:
            if self.verbose:
                print("structure found")
            for (x,y,w,h) in structures:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            return structures, img
        else:
            return [], img

class Face_Detector(Haar_Detector):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path,haar_face_file)
        Haar_Detector.__init__(self, detector_xml=file_path)