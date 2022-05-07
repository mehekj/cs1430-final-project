import math
import cv2
import numpy as np
from skimage.color import rgb2gray
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

DEGREE = np.pi/180

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
        return np.array([x0, y0])
    return None


# gets the intersection points of all the lines
def get_intersection_points(lines):
    intersections = []
    for line1 in lines:
        for line2 in lines:
            if not (line1[0] == line2[0] and line1[1] == line2[1]):
                point = get_line_intersection(line1, line2)
                if point is not None:
                    intersections.append(point)
    return np.array(intersections)


"""classify lines as horizontal or vertical """
def separate_lines(lines, threshold = DEGREE * 20):
    horizontals = []
    verticals = []
    for line in lines:
        theta = line[1]
        if theta <= threshold/2 or theta - DEGREE*180 >= -threshold/2:
            verticals.append(line)
        elif abs(theta - DEGREE*90) <= threshold:
            horizontals.append(line)
    return np.array(horizontals + verticals), np.array(horizontals), np.array(verticals)


def combine_lines(lines, rho_threshold=50, theta_threshold=np.pi / 3):
    best_lines = np.zeros((18, 2))
    count = 0
    for i in range(lines.shape[0]):
        rho = lines[i][0]
        theta = lines[i][1]
        if i == 0:
            best_lines[count] = lines[i]
            count += 1
        else:
            if rho < 0:
                rho *= -1
            closeness_rho = np.isclose(rho, best_lines[0:count, 0], atol=rho_threshold)
            closeness_theta = np.isclose(theta, best_lines[0:count, 1], atol=theta_threshold)
            closeness = np.all([closeness_rho, closeness_theta], axis=0)
            if not any(closeness) and count < best_lines.shape[0]:
                best_lines[count] = lines[i]
                count += 1

    return best_lines
    


def cluster_points(intersections):
    features = np.array(intersections)
    kmeans = MiniBatchKMeans(n_clusters=81, max_iter=500).fit(features)
    return np.ndarray.tolist(kmeans.cluster_centers_)


def get_lines(img):
    # convert to grayscale, blur, then get canny edges
    img = np.uint8(255 * rgb2gray(img))
    img = cv2.blur(img, (5, 5))
    edges = cv2.Canny(img, 50, 200)

    # get hough transform lines
    lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 125, None)
    return np.squeeze(lines, axis=1)


def plot_lines(img, lines, color=(255, 0, 0)):
    img = np.array(img)
    for i in range(0, len(lines)):
        rho = lines[i][0]
        theta = lines[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)
    
    return img


def plot_points(img, points):
    for i in points:
        x0 = i[0]
        y0 = i[1]
        
        img = np.array(img)
        cv2.circle(img, (int(x0), int(y0)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, (int(x0), int(y0)), radius=5, color=(0, 0, 255), thickness=2)
    
    return img


def get_board_corners(img):
    lines = get_lines(img)
    if lines is not None:
        lines = combine_lines(lines)
        intersections = get_intersection_points(lines)
        intersections = np.array(list(filter(lambda point : point[0] >= 0 and point[0] < img.shape[1] and point[1] >= 0 and point[1] < img.shape[0], intersections)))
        # corners = cluster_points(intersections)
        plot = plot_points(plot_lines(img, lines), intersections)
        return plot
    return None