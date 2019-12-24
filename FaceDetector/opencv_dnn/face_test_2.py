import numpy as np 
import cv2
from datetime import datetime
from cv_utilities import rectangler
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport"
rtsp_address = "rtsp://admin:kuassa-dev@192.168.100.199:554/onvif1"

#PARENT_FOLDER = "/home/bram/Documents/programming/bram_playground/FaceDetector/opencv_dnn/"
#PARENT_FOLDER = "/mnt/HDD/Docucap = cv2.VideoCapture(0)ments/programming/bram_playground/FaceDetector/opencv_dnn/"
PARENT_FOLDER = ""
PROTOTXT_FILENAME = PARENT_FOLDER +"deploy.prototxt.txt"
WEIGHTS = PARENT_FOLDER+"res10_300x300_ssd_iter_140000.caffemodel"

CONFIDENCE_THR = 0.25
class face_detector_opencv_dnn:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILENAME, WEIGHTS)
    
    def detect_faces(self, image):
        blob = cv2.dnn.blobFromImage(
            image = image, 
            scalefactor = 1.0,
            size = (300, 300), 
            mean = (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()
        return detections
        
    def detect_and_mark(self, image, confidence_thrs = CONFIDENCE_THR):
        (h,w) = image.shape[:2]
        detections = self.detect_faces(image)
        detected_faces_num = detections.shape[2]

        for i in range(0,detected_faces_num):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_thrs:
                box = detections[0 ,0 , i, 3:7]* np.array([w, h, w, h])
                (x_start, y_start, x_end, y_end) = box.astype("int")
                x_start, y_start, x_end, y_end = rectangler(x_start, y_start, x_end, y_end, w, h)
                text = "{:.2f}%".format(confidence*100)
                y = y_start - 10 if y_start -10 > 10 else y_start +10
                cv2.rectangle(image, (x_start, y_start),(x_end, y_end),(0,0,255),2)
                cv2.putText(image, text, (x_start,y),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,255),2)
                file_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+"_"+str(i)+"_NoCrop.png"
                print_succeed = cv2.imwrite(file_name,image)
                print(print_succeed)
                
        return image

    def detect_and_write(self, image, show = False, confidence_thrs = CONFIDENCE_THR):
        (h,w) = image.shape[:2]
        detections = self.detect_faces(image)
        detected_faces_num = detections.shape[2]

        for i in range(0,detected_faces_num):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_thrs:
                box = detections[0 ,0 , i, 3:7]* np.array([w, h, w, h])
                (x_start, y_start, x_end, y_end) = box.astype("int")
                x_start, y_start, x_end, y_end = rectangler(x_start, y_start, x_end, y_end, w, h)
                file_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+"_"+str(i)+".png"
                crop_img = image[y_start:y_end,x_start:x_end]
                print_succeed = cv2.imwrite(file_name,crop_img)
                print(print_succeed)
                if show:
                    text = "{:.2f}%".format(confidence*100)
                    y = y_start - 10 if y_start -10 > 10 else y_start +10
                    cv2.rectangle(image, (x_start, y_start),(x_end, y_end),(0,0,255),2)
                    cv2.putText(image, text, (x_start,y),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,255),2)

        return image

face_det = face_detector_opencv_dnn()
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(rtsp_address)

while(True):
    ret, frame = cap.read()
    if ret:
        result = face_det.detect_and_mark(image=frame)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result',result)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
