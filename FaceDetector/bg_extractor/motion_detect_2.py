import numpy as np
import cv2 
import os
import time
from datetime import datetime
from bg_extractor import *

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport"
rtsp_address = "rtsp://admin:kuassa-dev@192.168.100.199:554/onvif1"
#rtsp_address = 0
ret_false_max = 30
sleep_time = 0.0

class motion_detect_2:
    def __init__(self, output_dir = None, verbose = False, frame_block = 30, delay = 1):
        self.verbose = verbose
        self.frame_block = frame_block

        if output_dir == None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir
            if not os.path.isdir(self.output_dir):
                self.output_dir = os.getcwd()
        self.delay = delay
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
            self.bg_extractor = bg_extractor_2(image_shape =frame.shape, show_images=self.verbose)
            self.check_and_create_output_dir()
            
            if self.verbose: 
                print("saving images at :"+str(self.cur_dir))

    def check_and_create_output_dir(self):   
        #out_date_folder = datetime.now().strftime("%Y%m%d")

        out_date_folder = datetime.now().strftime("%d")
        self.cur_dir = os.path.join(self.output_dir, out_date_folder)
        if not os.path.exists(self.cur_dir):
            os.mkdir(self.cur_dir)

    def save_image(self, image, image_quality = 80):
        self.check_and_create_output_dir()

        filename = datetime.now().strftime("%H%M%S")+".jpg"
        full_filename = os.path.join(self.cur_dir, filename)  
        cv2.imwrite(full_filename, image, [cv2.IMWRITE_JPEG_QUALITY, image_quality])

    def process(self):
        while True:
            ret_false = 0

            #adding images to a list
            images = []
            frame_count = 0
            
            #obtaining images from camera
            while(self.cap.isOpened()):
                ret, frame = self.cap.read()
                if ret:
                    images.append(frame)
                    time.sleep(sleep_time)
                else:
                    ret_false += 1

                frame_count += 1
                if frame_count >= self.frame_block:
                    break
            
            if ret_false >= ret_false_max:
                self.cap.release()
                self.init_camera()
                ret_false = 0

            detected_images = []
            for image in images:
                result, _ = self.bg_extractor.find_moving_objects(image)
                if result:
                    detected_images.append(image)

            for image in detected_images:
                self.save_image(image)

            if self.verbose:
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

if __name__ == "__main__":
    main = motion_detect_2(verbose=False)
    main.process()
