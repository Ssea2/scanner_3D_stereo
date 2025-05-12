import numpy as np
import cv2 as cv
import glob


# mettre l'objet a 1.7m 

def calcam(img_folder):
        # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img_size = None

    images = glob.glob(img_folder)
    print(len(images))
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size=gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,5), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            #print("\n\n",corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,5), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    
        cv.destroyAllWindows()
    ### A TESTER transformer fx fy en mm 
    """print(gray.shape)
    print("mtx\n",mtx)
    print("dist\n",dist)
    print("rvecs\n",rvecs)
    print("tvecs\n",tvecs)"""
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx,dist

def calstereo(mtx1,mtx2,dist1,dist2, folder_cam_left, folder_cam_right):
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    objp = np.zeros((5*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right= []
    img_size = None

    images_left = glob.glob(folder_cam_left)
    print(len(images_left))

    images_right = glob.glob(folder_cam_right)
    for Lfname, Rfname in zip(images_left,images_right):
        # cam1
        Limg = cv.imread(Lfname)
        Lgray = cv.cvtColor(Limg, cv.COLOR_BGR2GRAY)

        #cam2
        Rimg = cv.imread(Lfname)
        Rgray = cv.cvtColor(Limg, cv.COLOR_BGR2GRAY)

        img_size=Lgray.shape[::-1]


        # Find the chess board corners
        ret, Lcorners = cv.findChessboardCorners(Lgray, (7,5), None)
        ret, Rcorners = cv.findChessboardCorners(Lgray, (7,5), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(Lgray,Lcorners, (11,11), (-1,-1), criteria)
            imgpoints_left.append(corners2)

            corners1 = cv.cornerSubPix(Rgray,Rcorners, (11,11), (-1,-1), criteria)
            imgpoints_right.append(corners1)

            #print("\n\n",corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(Limg, (7,5), corners2, ret)
            cv.imshow('img', Limg)
            cv.waitKey(500)

            cv.drawChessboardCorners(Rimg, (7,5), corners1, ret)
            cv.imshow('img', Rimg)
            cv.waitKey(500)
    
        cv.destroyAllWindows()

    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
    mtx2, dist2, img_size, criteria = criteria, flags = stereocalibration_flags)

    return R,T

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

"""mtx, dist = calcam("img_webcam/*.jpg")
print(mtx,dist)
img = cv.imread('img_webcam/img_119.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)"""

mtx1, dist1 = calcam("webcam_L/*.jpg")
mtx2, dist2 = calcam("webcam_R/*.jpg")
R,T = calstereo(mtx1,mtx2,dist1,dist2,"webcam_L/*.jpg","webcam_R/*.jpg")
print("stereo Rotation",R)
print("sterao translation",T)

#RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtx2 @ RT2 #projection matrix for C2