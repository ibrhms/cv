import cv2 
import numpy as np
import os

import haar_face_detector

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport"
rtsp_address = "rtsp://admin:kuassa-dev@192.168.100.199:554/onvif1"

if __name__ == "__main__":

    face_det = Face_Detector()
    cap = cv2.VideoCapture(rtsp_address)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)
    #cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        if ret:
            result,_ = face_det.find_and_draw_structure(img=frame)
            cv2.imshow('result',result)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
