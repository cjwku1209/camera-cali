import cv2
import numpy as np
img_top = cv2.imread('data/front.JPG')
img_bottom = cv2.imread('data/back.JPG')
img_top_grey = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
img_bottom_grey = cv2.cvtColor(img_bottom, cv2.COLOR_BGR2GRAY)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp1 = np.zeros((9*7,3), np.float32)
objp1[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2) 

# Arrays to store object points and image points from all the images.
objpoints1 = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.

ret1, corners1 = cv2.findChessboardCorners(img_top_grey, (7,9),None)
# If found, add object points, image points (after refining them)
print(ret1)
if ret1 == True:
    objpoints1.append(objp1)
    cv2.cornerSubPix(img_top_grey, corners1,(11,11),(-1,-1),criteria)
    imgpoints1.append(corners1)

    # Draw and display the corners
#     cv2.drawChessboardCorners(img_top, (7,6), corners, ret)
#     cv2.imshow('img',img_top)
#     cv2.waitKey(500)

# cv2.destroyAllWindows()
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1, imgpoints1, img_top_grey.shape[::-1],None,None)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp2 = np.zeros((9*7,3), np.float32)
objp2[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2) 

# Arrays to store object points and image points from all the images.
objpoints2 = [] # 3d point in real world space
imgpoints2 = [] # 2d points in image plane.

ret2, corners2 = cv2.findChessboardCorners(img_bottom_grey, (7,9),None)
# If found, add object points, image points (after refining them)
print(ret2)
if ret2 == True:
    objpoints2.append(objp2)
    cv2.cornerSubPix(img_bottom_grey, corners2,(11,11),(-1,-1),criteria)
    imgpoints2.append(corners2)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2, imgpoints2, img_bottom_grey.shape[::-1],None,None)
obj = objp1 + objp2
cv2.stereoCalibrate(obj, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, img_bottom_grey.shape[::-1])

