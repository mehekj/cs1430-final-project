from tkinter import filedialog
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import imgprocess
import game
from chessboard import display
import chess

'''
RUN THIS FILE
handles IMG CAPTURE and USER CONTROLS
'''
  
# define a video capture object
vid = cv2.VideoCapture(0)
filled_plot = None
line_plot = None
edges = None
my_board = game.Board()
board = chess.Board()
display.start(board.fen())
processor = imgprocess.ImageProcess()
  
while not display.checkForQuit():
    _, frame = vid.read() # read live video feed

    if frame is None:
        print("no camera input")
        break

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
    if line_plot is not None:
        new_frame = np.hstack((new_frame, line_plot))
    if filled_plot is not None:
        new_frame = np.hstack((new_frame, filled_plot))
        
    # displays frame
    cv2.imshow('live', new_frame)
    
    key = cv2.waitKey(1)
    # quits
    if key == ord('q'):
        break
    # captures photo and gets processed plot
    elif key == ord(' '):
        filled_board, captured_x, captured_y, filled_plot, line_plot = processor.get_board_state(frame)
    # this should return a FEN string
        FEN = my_board.get_fen_for_new_state(filled_board, captured_x, captured_y)
        print(board.fen())
    # this should update based on the returned FEN
        board.set_board_fen(FEN)

    # save img
    elif key == ord('s'):
        cv2.imwrite('savetest.jpg', cv2.resize(frame, (720, 720)))

    display.update(board.fen())

# quit program release cap obj and destroy windows
vid.release()
cv2.destroyAllWindows()
display.terminate()