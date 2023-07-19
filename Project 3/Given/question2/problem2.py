import numpy as np
import glob

import cv2 as cv 

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_x = 9
corners_y = 6

square_size = 21.5

checker_board_world = np.zeros((corners_x*corners_y, 3), np.float32)

checker_board_world[:, :2] = np.mgrid[0:corners_x, 0: corners_y]. T.reshape(-1, 2)*square_size

# print(checker_board_world)

checker_board_world_points = []
checker_board_image_points = []

# image1 = '/home/ab/enpm673/project3/question2/1.jpg'
# image2 = '/home/ab/enpm673/project3/question2/2.jpg'
# image3 = '/home/ab/enpm673/project3/question2/3.jpg'
# image4 = '/home/ab/enpm673/project3/question2/4.jpg'
# image5 = '/home/ab/enpm673/project3/question2/5.jpg'
# image6 = '/home/ab/enpm673/project3/question2/6.jpg'
# image7 = '/home/ab/enpm673/project3/question2/7.jpg'
# image8 = '/home/ab/enpm673/project3/question2/8.jpg'
# image9 = '/home/ab/enpm673/project3/question2/9.jpg'
# image10 = '/home/ab/enpm673/project3/question2/10.jpg'
# image11 = '/home/ab/enpm673/project3/question2/11.jpg'
# image12 = '/home/ab/enpm673/project3/question2/12.jpg'
# image13 = '/home/ab/enpm673/project3/question2/13.jpg'

# images_list = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, image13]
images_list =  glob.glob('/home/ab/enpm673/project3/question2/*.jpg')

for i in images_list:

    img = cv.imread(i)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (corners_x, corners_y), None)

    if ret == True:

        checker_board_world_points.append(checker_board_world)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        checker_board_image_points.append(corners2)

        cv.drawChessboardCorners(img, (corners_x,corners_y), corners2, ret)

        cv.namedWindow('corners', cv.WINDOW_NORMAL)
        cv.resizeWindow('corners', img.shape[1], img.shape[0])
        cv.imshow('corners', img)
        cv.waitKey(1000)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif cv.getWindowProperty('corners', cv.WND_PROP_VISIBLE) < 1:
        break
cv.destroyAllWindows()    

ret, camera_matrix, distortion_coefficient, rotation_vector, traslation_vector = cv.calibrateCamera(checker_board_world_points, checker_board_image_points, gray.shape[::-1], None, None)

print("Camera intrinsic matrix = \n" ,camera_matrix)
print("\n")

reprojection_error = []

# print(len(checker_board_world_points))

for i in range(len(checker_board_image_points)):
    checker_board_image_points2 , _ = cv.projectPoints(checker_board_world_points[i], rotation_vector[i], traslation_vector[i], camera_matrix, distortion_coefficient)
    error = cv.norm(checker_board_image_points[i], checker_board_image_points2, cv.NORM_L2)/len(checker_board_image_points2)
    print("Reprojection error for image  =", error)
    reprojection_error.append(error)

print("\n")

mean = np.mean(reprojection_error)

print("Mean reprojection error = ",mean)
print("\n")






