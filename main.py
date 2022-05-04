import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import imgprocess
import game

'''
RUN THIS FILE
handles IMG CAPTURE and USER CONTROLS
'''
  
# define a video capture object
vid = cv2.VideoCapture(0)
last_cap = None
edges = None
num_caps = 10
captured_frames = []
cap_count = -1
  
while(True):
    # read live video feed
    _, frame = vid.read()
    # frame = np.flip(io.imread('test.jpeg'), axis=-1) # google image
    # frame = np.flip(io.imread('savetest.jpg'), axis=-1) # our board

    # crop to square
    if frame.shape[1] > frame.shape[0]:
        cropx = int((frame.shape[1] - frame.shape[0])/2)
        cropy = 0
        frame = frame[cropy:frame.shape[0] - cropy, cropx:frame.shape[1] - cropx]
    elif frame.shape[0] > frame.shape[1]:
        cropx = 0
        cropy = int((frame.shape[0] - frame.shape[1])/2)
        frame = frame[cropy:frame.shape[0] - cropy, cropx:frame.shape[1] - cropx]
  
    # adds processed captured photo to window next to live cam feed
    new_frame = frame
    if last_cap is not None:
        new_frame = np.hstack((frame, last_cap))
        
    # displays frame
    cv2.imshow('live', new_frame)
    
    key = cv2.waitKey(1)
    # quits
    if key == ord('q'):
        break
    # captures photo and gets processed plot
    elif key == ord(' ') and cap_count == -1:
        # last_cap = imgprocess.get_lines(frame)
        captured_frames = []
        cap_count = 0
    # save img
    elif key == ord('s'):
        cv2.imwrite('savetest.jpg', cv2.resize(frame, (720, 720)))
    
    if cap_count != -1 and cap_count < num_caps:
        captured_frames.append(frame)
        cap_count += 1
        if cap_count == num_caps:
            last_cap = imgprocess.get_board_corners(captured_frames)
            captured_frames = []
            cap_count = -1

# quit program release cap obj and destroy windows
vid.release()
cv2.destroyAllWindows()