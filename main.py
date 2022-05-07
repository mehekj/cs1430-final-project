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
  
while(True):
    # read live video feed
    # _, frame = vid.read()
    # frame = np.flip(io.imread('test.jpeg'), axis=-1) # google image
    frame = np.flip(io.imread('savetest.jpg'), axis=-1) # our board

    # crop to square
    cropx = 0
    cropy = 0
    if frame.shape[1] > frame.shape[0]:
        cropx = int((frame.shape[1] - frame.shape[0])/2)
        cropy = 0
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
    elif key == ord(' '):
        last_cap = imgprocess.get_lines(frame)
    # save img
    elif key == ord('s'):
        cv2.imwrite('savetest.jpg', cv2.resize(frame, (720, 720)))

# quit program release cap obj and destroy windows
vid.release()
cv2.destroyAllWindows()