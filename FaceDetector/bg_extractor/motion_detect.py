import numpy as np
import cv2 
import os
import time
import argparse
from datetime import datetime
from bg_extractor import *

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport"
rtsp_address = "rtsp://admin:kuassa-dev@192.168.100.199:554/onvif1"
#rtsp_address = 0
ret_false_max = 30
sleep_time = 200

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="output directory", type = str)

class motion_detect_1:
    def __init__(self, verbose = False,output_dir = None):
        self.show_images = verbose

        if output_dir == None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir
            if not os.path.isdir(self.output_dir):
                self.output_dir = os.getcwd()
                
        self.init_camera()

    def init_camera(self):
        cap_is_opened = False
        
        while not cap_is_opened:
            self.cap = cv2.VideoCapture(rtsp_address)
            cap_is_opened = self.cap.isOpened()
            if not cap_is_opened:
                self.cap.release()
            
        is_cap = False
        while not is_cap:
            is_cap, _ = self.cap.read()

        if is_cap:
            _, frame = self.cap.read()
            self.bg_extractor = bg_extractor_1(image_shape =frame.shape, show_images=self.show_images)
            self.check_and_create_output_dir()
            print("saving images at :"+str(self.cur_dir))


    def check_and_create_output_dir(self):   

        out_date_folder = datetime.now().strftime("%m%d")
        self.cur_dir = os.path.join(self.output_dir, out_date_folder)
        if not os.path.exists(self.cur_dir):
            os.mkdir(self.cur_dir)

    def save_image(self, image):
        self.check_and_create_output_dir()

        filename = datetime.now().strftime("%H%M%S%f")[:-5]+".jpg"
        full_filename = os.path.join(self.cur_dir, filename)  
        cv2.imwrite(full_filename, image, [cv2.IMWRITE_JPEG_QUALITY, 99])

    def process(self):
        ret_false = 0
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            if ret:
                result, _ = self.bg_extractor.find_moving_objects(frame)
                if result:   
                    self.save_image(frame)

                k = cv2.waitKey(sleep_time) & 0xff
                if k == 27:
                    break
            else:
                ret_false += 1
                if ret_false >= ret_false_max:
                    self.cap.release()
                    self.init_camera()
                    ret_false = 0
           
        self.cap.release()

if __name__ == "__main__":
    args = parser.parse_args()
    main = motion_detect_1(verbose=True, output_dir=args.dir)
    main.process()
