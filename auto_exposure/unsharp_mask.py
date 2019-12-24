
import cv2
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc

from skimage import data
from skimage.filters import unsharp_mask

from skimage.filters import threshold_yen, threshold_mean, threshold_minimum
from skimage.exposure import rescale_intensity

from skimage.util import img_as_float, img_as_ubyte

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the video file")
ap.add_argument('-o', "--output", help="path to output video file")
args = vars(ap.parse_args())
unsharped_mask_v(args['input'],args['output'],True)

def is_img_same(in1, in2):
    difference = in1 - in2
    b, g, r = cv2.split(difference)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    return False

def unsharped_mask_v(self, video_input, video_output, verbose = False): 
    input_video = VideoCapture(video_input)       
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    print('fps : {}'.format(fps))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('height : {}'.format(height))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('width : {}'.format(width))
    out_video = VideoWriter(video_output, VideoWriter_fourcc(*'XVID'), fps, (width, height))

    _,past_frame = input_video.read()

    while True:
        _,frame = input_video.read()
        if frame is None: 
            break
        if verbose:
            cv2.imshow('input', frame)
        if is_img_same(past_frame, frame):
            if verbose:
                print('images are same')
        else:
            #print('images are not the same')

            past_frame = frame.copy()
            #converting to skimage
            frame = img_as_float(frame)
            frame = unsharp_mask(frame,radius=20)
            frame = img_as_ubyte(frame)
            if verbose:
                cv2.imshow('processed', frame)
            out_video.write(frame)
        if verbose:
            cv2.waitKey(1)
            
    input_video.release()
    out_video.release()
    
def unsharped_mask_im(self, images_input, verbose = False): 
    output=[]

    past_frame = images_input[0].copy()

    for i, frame in enumerate(images_input):
        if verbose:
            cv2.imshow('input', frame)
        if is_img_same(past_frame, frame):
            if verbose:
                print('images are same')
        else:
            #print('images are not the same')

            past_frame = frame.copy()
            #converting to skimage
            frame = img_as_float(frame)
            frame = unsharp_mask(frame,radius=20)
            frame = img_as_ubyte(frame)
            if verbose:
                cv2.imshow('processed', frame)
            output.append(frame)
        if verbose:
            cv2.waitKey(1)
    return output


    