import cv2
import numpy as np
img_top = cv2.imread('data/chessboard_normal.png')
img_bottom = cv2.imread('data/chessboard_thermal.png')

img_top_grey = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
img_bottom_grey = cv2.cvtColor(img_bottom, cv2.COLOR_BGR2GRAY)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp1 = np.zeros((6*8,3), np.float32)
objp1[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints1 = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.

ret, corners = cv2.findChessboardCorners(img_top_grey, (6,8),flags = cv2.CALIB_CB_ADAPTIVE_THRESH)
# If found, add object points, image points (after refining them)
print(ret)
# if ret == True:
#     objpoints1.append(objp1)
#     cv2.cornerSubPix(img_top_grey, corners,(11,11),(-1,-1),criteria)
#     imgpoints1.append(corners)
#
#     # Draw and display the corners
#     cv2.drawChessboardCorners(img_top, (6,8), corners, ret)
#     cv2.imshow('img',img_top)
#     cv2.waitKey(100000)

# cv2.imshow('img',img_bottom_grey)
# cv2.waitKey(100000)

equ = cv2.equalizeHist(img_bottom_grey)
# cv2.imshow('img',equ)
# cv2.waitKey(100000)

cv2.imshow('img',img_bottom_grey)
# cv2.waitKey(100000)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img_bottom_grey)
# cv2.imshow('img',cl1)
# cv2.waitKey(100000)

# mser = cv2.MSER()
# mser_areas = mser.detect()

vis = img_bottom_grey.copy()
mser= cv2.MSER_create()
regions = mser.detectRegions(img_bottom_grey, None)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()