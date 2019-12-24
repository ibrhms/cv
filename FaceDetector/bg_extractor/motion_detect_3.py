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
sleep_time = 100

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="output directory", type = str)

class motion_detect_3:
    def __init__(self, verbose = False,output_dir = None):
        self.show_images = verbose
        self.image_container = []
        self.video_name = None
        self.fps = int(1000/sleep_time)
        if output_dir == None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir
            if not os.path.isdir(self.output_dir):
                self.output_dir = os.getcwd()
                
        self.init_camera()
        self.record_mode = False

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
            self.frame_shape = frame.shape
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

    def save_video(self):
        if len(self.image_container) > 5: 
            self.check_and_create_output_dir()
            filename = datetime.now().strftime("%H%M%S%f")[:-5]+".avi"
            full_filename = os.path.join(self.cur_dir, filename) 
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vid_writer = cv2.VideoWriter(full_filename,fourcc, self.fps, (self.frame_shape[1],self.frame_shape[0]))
        
            for image in self.image_container:
                vid_writer.write(image)

            vid_writer.release()

    def process(self):
        ret_false = 0
        not_detected_count = self.fps * 2
        detected_count = 0
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            if ret:
                result, _ = self.bg_extractor.find_moving_objects(frame)
                if result:   
                    self.record_mode = True
                    self.image_container.append(frame)
                    not_detected_count = self.fps * 2
                    detected_count +=1
                else:
                    if not_detected_count > 0:
                        not_detected_count -= 1
                        if (not_detected_count <= 0) and (detected_count > 0):
                            self.save_video()
                            self.image_container = []
                            detected_count = 0
                            not_detected_count = 0

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
    main = motion_detect_3(verbose=False, output_dir=args.dir)
    main.process()
