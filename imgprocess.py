import math
import cv2
import numpy as np
from skimage.color import rgb2gray

def get_line_intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def get_intersection_points(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(get_line_intersection(line1, line2)) 
    return intersections

'''
handles the actual computer vision part
'''

def get_lines(img):
    # convert to grayscale, blur, then get canny edges
    img = np.uint8(255 * rgb2gray(img))
    img = cv2.blur(img, (5, 5))
    edges = cv2.Canny(img, 50, 200)

    # get hough transform lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, None)

    # plots hough transform lines over blurred img for testing
    plot = np.stack((img,) * 3, axis=-1)
    xys = []
    if lines is not None:
        intersections = get_intersection_points(lines)
        print(intersections)
        for i in intersections:
            x0 = intersections[0][0]
            y0 = intersections[0][1]
            cv2.circle(plot, (x0, y0), radius=2, color=(0, 0, 255), thickness=-1)


        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            xys.append([pt1, pt2])
            cv2.line(plot, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    return plot