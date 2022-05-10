import math
import cv2
from cv2 import transform
import numpy as np
from skimage.color import rgb2gray
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

DEGREE = np.pi/180

'''
handles the actual computer vision part
'''


'''
first click: take calibration img of empty board
do full calculations -> lines, points, normalization, center point etc.
store a matrix of points used for comparison
second click: full board setup
do full calculations
get difference of points vs calibration
rank diffs and take top 32 and define threshold based on that

every image after
do full calculations
get diffs
if abs(diff) is above threshold that cell is filled
based on pos or neg threshold set color
return matrix of 0,1,2 to main.py to pass to game
'''

class ImageProcess:
    def __init__(self):
        self.calibration_feats = None
        self.diff_threshold = None
        self.calibration_img = None
        self.last_avg_intensity = None
        self.last_num_piece = 32

    # gets intersections between two lines
    def get_line_intersection(self, line1, line2):
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
    def get_intersection_points(self, horizontals, verticals):
        intersections = []
        for line1 in horizontals:
            for line2 in verticals:
                point = self.get_line_intersection(line1, line2)
                if point is not None:
                    intersections.append(point)
        return np.array(intersections)

    # separates lines into horizontal and vertical lines, also filters out diagonals
    def separate_lines(self, lines, threshold = DEGREE * 20):
        horizontals = []
        verticals = []
        for line in lines:
            theta = line[1]
            if theta <= threshold/2 or theta - DEGREE*180 >= -threshold/2:
                verticals.append(line)
            elif abs(theta - DEGREE*90) <= threshold:
                horizontals.append(line)
        return np.array(horizontals + verticals), np.array(horizontals), np.array(verticals)

    # takes the best 18 distinct lines from many overlapping
    def combine_lines(self, lines, n, rho_threshold=30, theta_threshold=np.pi / 6):
        best_lines = np.zeros((n, 2))
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
                    lines[i][0] *= -1
                    theta -= np.pi
                    lines[i][1] -= np.pi
                closeness_rho = np.isclose(rho, best_lines[0:count, 0], atol=rho_threshold)
                closeness_theta = np.isclose(theta, best_lines[0:count, 1], atol=theta_threshold)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and count < best_lines.shape[0]:
                    best_lines[count] = lines[i]
                    count += 1

        return best_lines

    # uses kmeans to find corner cluster centers (did not work)
    def cluster_points(self, intersections):
        features = np.array(intersections)
        kmeans = MiniBatchKMeans(n_clusters=81, max_iter=500).fit(features)
        return np.ndarray.tolist(kmeans.cluster_centers_)

    # gets hough lines from image
    def get_lines(self, img):
        # convert to grayscale, blur, then get canny edges
        img = np.uint8(255 * rgb2gray(img))
        img = cv2.blur(img, (5, 5))
        edges = cv2.Canny(img, 50, 200)

        # get hough transform lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100, None)
        return np.squeeze(lines, axis=1)

    # plots lines on image
    def plot_lines(self, img, lines, color=(255, 0, 0)):
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

    # plots points on image
    def plot_points(self, img, points):
        for i in points:
            x0 = i[0]
            y0 = i[1]
            
            img = np.array(img)
            cv2.circle(img, (int(x0), int(y0)), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(img, (int(x0), int(y0)), radius=5, color=(0, 0, 255), thickness=2)
        
        return img

    # puts found board corners into ordered grid matrix
    def get_intersection_matrix(self, points):
        points = points[points[:, 1].argsort()]
        matrix = np.reshape(points, (9, 9, 2))
        for i in range(matrix.shape[0]):
            matrix[i, :, :] = matrix[i, :, :][matrix[i, :, :][:, 0].argsort()]
        return matrix

    # warps board to square using corners
    def warp_image(self, img, matrix):
        orig_coords = np.float32([matrix[0, 0, :], matrix[0, -1, :], matrix[-1, 0, :], matrix[-1, -1, :]])
        new_coords = np.float32([[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]])
        transform_mat = cv2.getPerspectiveTransform(orig_coords, new_coords)
        warped_img = cv2.warpPerspective(img, transform_mat, img.shape[:2])
        
        #just for a sanity check, checking to see where the matrix points get mapped... not working atm
        warped_mtx = np.zeros((9, 9, 2))
        for i in range(9):
            for j in range(9):
                p = matrix[i, j]
                warped_mtx[i, j, 0] = int((transform_mat[0][0] * p[0] + transform_mat[0][1] * p[1] + transform_mat[0][2]) / ((transform_mat[2][0] * p[0] + transform_mat[2][1] * p[1] + transform_mat[2][2])))
                warped_mtx[i, j, 1] = int((transform_mat[1][0] * p[0] + transform_mat[1][1] * p[1] + transform_mat[1][2]) / ((transform_mat[2][0] * p[0] + transform_mat[2][1] * p[1] + transform_mat[2][2])))
        
        return warped_img, warped_mtx


    def get_features(self, img, matrix):
        gray = np.uint8(255 * rgb2gray(img))
        normalized = np.zeros(gray.shape)
        normalized = cv2.normalize(gray, normalized, 0, 255, cv2.NORM_MINMAX)
        features = np.zeros((8, 8))
        avg_intensities = np.zeros((8, 8, 3))
        # plot = np.stack((normalized,) * 3, axis=-1)
        for i in range(8):
            for j in range(8):
                p1 = matrix[i, j, :]
                p2 = matrix[i, j+1, :]
                p3 = matrix[i+1, j, :]
                p4 = matrix[i+1, j+1, :]
                center = np.uint32(np.average([p1, p2, p3, p4], axis=0))

                feat_sz = np.uint32(0.55 * (np.abs(p1 - center) + np.abs(p2 - center) + np.abs(p3 - center) + np.abs(p4 - center)) // 4)

                area = normalized[center[0] - feat_sz[0] : center[0] + feat_sz[0], center[1] - feat_sz[1] : center[1] + feat_sz[1]]
                blurred = cv2.blur(area, (3, 3))
                edges = cv2.Canny(blurred, 50, 100)

                features[i, j] = np.sum(edges)
                avg_intensities[i, j, 0] = np.average(img[center[0] - feat_sz[0] : center[0] + feat_sz[0], center[1] - feat_sz[1] : center[1] + feat_sz[1], 0])
                avg_intensities[i, j, 1] = np.average(img[center[0] - feat_sz[0] : center[0] + feat_sz[0], center[1] - feat_sz[1] : center[1] + feat_sz[1], 1])
                avg_intensities[i, j, 2] = np.average(img[center[0] - feat_sz[0] : center[0] + feat_sz[0], center[1] - feat_sz[1] : center[1] + feat_sz[1], 2])

        return features, avg_intensities



    def get_board_corners(self, img):
        lines = self.get_lines(img)
        if lines is not None:
            lines, horizontals, verticals = self.separate_lines(lines)
            horizontals = self.combine_lines(horizontals, 9)
            verticals = self.combine_lines(verticals, 9)
            # return self.plot_lines(self.plot_lines(img, verticals), horizontals)
            intersections = self.get_intersection_points(horizontals, verticals)
            intersections = np.array(list(filter(lambda point : point[0] >= 0 and point[0] < img.shape[1] and point[1] >= 0 and point[1] < img.shape[0], intersections)))
            intersection_matrix = self.get_intersection_matrix(intersections)
            return intersection_matrix
        return None


    def get_board_state(self, img):
        intersection_matrix = self.get_board_corners(img)
        # return intersection_matrix
        if intersection_matrix is not None:
            warped_img, warped_mtx = self.warp_image(img, intersection_matrix)
            print(warped_mtx)
            features, avg_intensities = self.get_features(warped_img, warped_mtx)
            if self.calibration_feats is None:
                self.calibration_feats = features
                plot = self.plot_points(warped_img, np.reshape(warped_mtx, (-1, 2)))
            elif self.diff_threshold is None:
                self.diff_threshold = self.calibrate_threshold(features)
                diffs = np.abs(self.calibration_feats - features)
                filled = np.zeros(features.shape)
                filled[diffs > self.diff_threshold] = 1
                plot = self.plot_squares(filled, warped_img, warped_mtx)
                self.last_num_piece = np.sum(filled)
                self.last_avg_intensity = avg_intensities
            else:
                diffs = np.abs(self.calibration_feats - features)
                filled = np.zeros(features.shape)
                filled[diffs > self.diff_threshold] = 1
                plot = self.plot_squares(filled, warped_img, warped_mtx)
                gray = np.uint8(255 * rgb2gray(warped_img))
                normalized = np.zeros(gray.shape)
                normalized = cv2.normalize(gray, normalized, 0, 255, cv2.NORM_MINMAX)
                captured_x, captured_y = self.check_piece_diff(filled, avg_intensities)
                # print(filled)
                print(captured_x, captured_y)
                self.last_num_piece = np.sum(filled)
                self.last_avg_intensity = avg_intensities
            return plot
        return None

    
    def plot_squares(self, filled, img, matrix):
        print(matrix)
        for i in range(8):
            for j in range(8):
                if filled[i, j] == 1:
                    cv2.rectangle(img, matrix[j, i], matrix[j + 1, i + 1], (0, 255, 0), thickness=2)
        return img


    def calibrate_threshold(self, features):
        # diffs = self.calibration_feats - features
        # diffs = np.sort(np.abs(diffs), axis=None)
        # return diffs[3]
        return 0

    def check_piece_diff(self, filled, curr_intensities):
        if np.sum(filled) < self.last_num_piece:
            max_diff = 0
            captured_x = -1
            captured_y = -1

            for i in range(8):
                for j in range(8):
                    if filled[i, j] == 1:
                        diff = np.sum(np.abs(curr_intensities[i, j] - self.last_avg_intensity[i, j]))
                        if diff > max_diff:
                            captured_x = i
                            captured_y = j
                            max_diff = diff

            return captured_x, captured_y

        else:
            return -1, -1
