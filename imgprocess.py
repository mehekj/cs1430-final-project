import math
import cv2
import numpy as np
from skimage.color import rgb2gray

'''
handles the actual computer vision part
'''

# gets intersections between two lines
def get_line_intersection(line1, line2):
    rho1 = line1[0]
    theta1 = line1[1]
    rho2 = line2[0]
    theta2 = line2[1]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if (np.linalg.det(A) != 0):
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]
    return None

# gets the intersection points of all the lines
def get_intersection_points(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    point = get_line_intersection(line1, line2)
                    if point is not None:
                        intersections.append(point)
    return intersections


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
        
        # filters out lines that aren't close enough
        intersections = filter(lambda point : 
            point[0][0] >= 0 and point[0][0] < plot.shape[1] and point[0][1] >= 0 and point[0][1] < plot.shape[0]
        , intersections)

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

        # plots the intersections
        for i in list(intersections):
            x0 = i[0][0]
            y0 = i[0][1]
            
            cv2.circle(plot, (int(x0), int(y0)), radius=10, color=(255, 0, 0), thickness=-1)
    return plot