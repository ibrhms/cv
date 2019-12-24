import cv2
import numpy as np

FNB = '24'
FILENAME = FNB+'.avi'
OUT_FILENAME = FNB +'-out.mp4'

UM_RAD = 9

def unsharp_mask(frame):
    gaussian = cv2.GaussianBlur(frame, (UM_RAD,UM_RAD), UM_RAD +1)
    unsharp_image = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0, frame)
    return unsharp_image


class frame_dropper:
    def __init__(self, threshold):
        self.threshold = threshold
        self.past_frame = None
        self.first_time = True
        self.num_of_pixel = 1
        self.norm = 0
    def process(self, new_frame):
        if (self.first_time):
            self.first_time = False
            self.past_frame = new_frame.copy()
            for i in range(len(new_frame.shape)):
                self.num_of_pixel *= new_frame.shape[i] 
            self.norm = 1. / (self.num_of_pixel * 255.)

            return True, self.past_frame
        
        #calculating diff
        # if (new_frame == None)
        #      return False, self.past_frame     

        diff = np.sum(np.absolute(self.past_frame - new_frame)) * self.norm
        
        #if frames are different
        if diff > self.threshold:
            self.past_frame = new_frame.copy()
            return True, self.past_frame
        else:
            return False, self.past_frame



if __name__ == "__main__":
    fd = frame_dropper(1e-8)
    cap = cv2.VideoCapture(FILENAME)
    _, frame = cap.read()
    out = cv2.VideoWriter(OUT_FILENAME,cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            ret2 , res = fd.process(frame)
            if (ret2):  
                #performing unsharp mask
                # copy_res = res.copy()                
                # res = unsharp_mask(res)
                # diff = np.abs(res -copy_res)
                cv2.imshow('res',res)
                out.write(res)
        else: 
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()