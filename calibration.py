import numpy as np
import cv2 as cv
import glob
import faulthandler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# mettre l'objet a 1.7m 

def calcam(img_folder, chessboad_size, square_size):
    """
    @param:
        -img_folder :str: chemin vers le dossier qui contient les images de calibration
    @return:
        -mtx :array: la matrice des parametres intrinsèque de la caméra
        -dist :array: la list des coeficients de distortion
    """
        # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboad_size[0]*chessboad_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboad_size[1],0:chessboad_size[0]].T.reshape(-1,2) * square_size
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img_size = None

    images = glob.glob(img_folder)
    print(len(images))

    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size=gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (chessboad_size[1],chessboad_size[0]), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            print(ret, fname, img.shape)
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            #print("\n\n",corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (chessboad_size[1],chessboad_size[0]), corners2, ret)
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



def calstereo(mtx1,mtx2,dist1,dist2, folder_cam_left, folder_cam_right, chessboad_size, square_size):
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    objp = np.zeros((chessboad_size[0]*chessboad_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboad_size[1],0:chessboad_size[0]].T.reshape(-1,2) *square_size
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right= []
    #img_size = None

    images_left = sorted(glob.glob(folder_cam_left))
    #print(images_left[0])

    images_right = sorted(glob.glob(folder_cam_right))
    #print(images_right[0])
    for Lfname, Rfname in zip(images_left,images_right):
        # cam1
        Limg = cv.imread(Lfname)

        #cam2
        Rimg = cv.imread(Rfname)
        if Rimg is None or Limg is None:
            print(f"Warning: Could not read images {Lfname} or {Rfname}")
            continue

        Lgray = cv.cvtColor(Limg, cv.COLOR_BGR2GRAY)
        Rgray = cv.cvtColor(Rimg, cv.COLOR_BGR2GRAY)
        img_size=Lgray.shape[::-1]


        # Find the chess board corners
        ret, Lcorners = cv.findChessboardCorners(Lgray, (chessboad_size[1],chessboad_size[0]), None)
        ret, Rcorners = cv.findChessboardCorners(Rgray, (chessboad_size[1],chessboad_size[0]), None)

        if Rcorners is None or Lcorners is None:
            print(f"Warning: pas trouver le damier images {Lfname} or {Rfname}")
            continue

        if Lcorners.all() == None:
            print("L: ",None)
        if Rcorners.all() == None:
            print("R: ",None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(Lgray,Lcorners, (11,11), (-1,-1), criteria)
            imgpoints_left.append(corners2)

            corners1 = cv.cornerSubPix(Rgray,Rcorners, (11,11), (-1,-1), criteria)
            imgpoints_right.append(corners1)

            #print("\n\n",corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(Limg, (chessboad_size[1],chessboad_size[0]), corners2, ret)
            cv.imshow('Left', Limg)
            cv.waitKey(500)

            cv.drawChessboardCorners(Rimg, (chessboad_size[1],chessboad_size[0]), corners1, ret)
            cv.imshow('Right', Rimg)
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

"""mtx, dist = calcam("images/calibration/img_webcam/*.jpg")
print(mtx,dist)
img = cv.imread('images/calibration/img_webcam/img_119.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)"""

#print(cv.imread("camera2/camera_2_image_20250512_143203.jpg").shape)
mtx1, dist1 = calcam("images/calibration/camera1_class/*.jpg", (5,7), 160)
print("\n\n\n\n\n\nMx1 ", mtx1)
mtx2, dist2 = calcam("images/calibration/camera2_class/*.jpg",(5,7), 160)
print("\n\n\n\n\n\nMx2 ", mtx2)
R,T = calstereo(mtx1,mtx2,dist1,dist2,"images/calibration/camera1_class/*.jpg","images/calibration/camera2_class/*.jpg",(5,7), 160)
# print("\n\n\n\n\n\nstereo Rotation",R)
# print("\n\n\n\n\n\n sterao translation",T)

# #RT matrix for C1 is identity. 
# RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
# P1 = mtx1 @ RT1 #projection matrix for C1
 
# #RT matrix for C2 is the R and T obtained from stereo calibration.
# RT2 = np.concatenate([R, T], axis = -1)
# P2 = mtx2 @ RT2 #projection matrix for C2"""

print(f"{np.linalg.norm(T):.2f}")
"""
print(img1.shape)
print(img2.shape)
# Initiate SIFT detector
sift = cv.SIFT_create()
 
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
 
flann = cv.FlannBasedMatcher(index_params,search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
 
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < n.distance:
        matchesMask[i]=[1,0]
 
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
 
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
 
cv.imshow("t",img3)
cv.waitKey(0)
cv.destroyAllWindows()


def match_features(img1_path, img2_path):
    # 1. Chargement des images
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    
    # Vérification du chargement correct des images
    if img1 is None or img2 is None:
        print("Erreur: Impossible de charger une ou les deux images")
        return
    
    # 2. Conversion en niveaux de gris
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    # 3. Détection des points d'intérêt et calcul des descripteurs
    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # Vérification si les descripteurs sont vides
    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
        print("Erreur: Impossible de trouver des descripteurs dans une ou les deux images")
        return
    
    # Vérification supplémentaire: les descripteurs doivent être du même type
    if descriptors1.dtype != np.float32:
        descriptors1 = np.float32(descriptors1)
    if descriptors2.dtype != np.float32:
        descriptors2 = np.float32(descriptors2)
    
    # 4. Matching des descripteurs
    # Utilisation de BFMatcher au lieu de FLANN pour plus de robustesse
    bf = cv.BFMatcher()
    
    # Utilisez le matcher approprié et avec gestion d'erreurs
    try:
        # Vérifiez si nous avons assez de descripteurs pour kNN avec k=2
        k = min(2, len(descriptors2))
        matches = bf.knnMatch(descriptors1, descriptors2, k=k)
        
        # 5. Filtrage des bons matchs avec le test de ratio de Lowe (uniquement si k=2)
        good_matches = []
        if k == 2:
            for pair in matches:
                if len(pair) == 2:  # S'assurer que nous avons bien deux matches
                    m, n = pair
                    if m.distance < 0.9* n.distance:
                        good_matches.append(m)
        else:
            # Si k=1, prenez simplement tous les matches
            good_matches = [m[0] for m in matches if len(m) > 0]
            
    except cv.error as e:
        print(f"Erreur lors du matching: {e}")
        # Solution alternative: utilisation de matcher.match() au lieu de knnMatch
        matches = bf.match(descriptors1, descriptors2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:30]  # Prendre les 30 meilleurs
    
    # 6. Affichage des résultats
    img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, 
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Conversion BGR->RGB pour matplotlib
    img_matches = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)
    
    cv.imshow("t",img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return keypoints1, keypoints2, good_matches

# Utilisation
key1,key2,matchs=match_features('images/test/sc3/screen1-1/camera_1_image_20250512_181245.jpg', 'images/test/sc3/screen2-1/camera_2_image_20250512_181245.jpg')

uvs1= []
uvs2 = []

for m in matchs:
    uvs1.append(key1[m.queryIdx].pt)
    uvs2.append(key2[m.trainIdx].pt)

print(uvs1)
print(uvs2)

p3ds = []
for uv2, uv1 in zip(uvs1, uvs2):
    _p3d = DLT(P1, P2, uv1, uv2)
    p3ds.append(_p3d)
p3ds = np.array(p3ds)

#print(p3ds)

faulthandler.enable()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-15, 5)
ax.set_ylim3d(-10, 10)
ax.set_zlim3d(10, 30)
 
connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
for _c in connections:
    print(p3ds[_c[0]])
    print(p3ds[_c[1]])
    ax.plot(xs = [p3ds[_c[0],0], p3ds[_c[1],0]], ys = [p3ds[_c[0],1], p3ds[_c[1],1]], zs = [p3ds[_c[0],2], p3ds[_c[1],2]], c = 'red')
 
plt.show()"""