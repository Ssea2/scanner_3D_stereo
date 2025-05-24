import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import time
from tabulate import tabulate

class StereoCalibration:
    def __init__(self, checkerboard_size=(9, 6), square_size=30):
        """
        Initialise la calibration stéréo
        checkerboard_size: (colonnes, lignes) de coins intérieurs
        square_size: taille d'un carré en mm (peut être arbitraire)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Préparation des points du damier
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Listes pour stocker les points
        self.objpoints = []  # Points 3D dans l'espace réel
        self.imgpoints_left = []  # Points 2D dans l'image gauche
        self.imgpoints_right = []  # Points 2D dans l'image droite
        
        # Paramètres de calibration
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation entre les caméras
        self.T = None  # Translation entre les caméras
        self.E = None  # Matrice essentielle
        self.F = None  # Matrice fondamentale
        
    def find_corners(self, img_path):
        """Trouve les coins du damier dans une image"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Trouve les coins du damier
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # Affine la position des coins
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Dessine les coins pour visualisation
            img_with_corners = cv2.drawChessboardCorners(img.copy(), self.checkerboard_size, corners, ret)
            return ret, corners, img_with_corners
        
        return ret, None, img
    
    def calibrate_cameras(self, left_images_path, right_images_path):
        """Calibre les deux caméras"""
        left_images = sorted(glob.glob(left_images_path))
        right_images = sorted(glob.glob(right_images_path))
        
        if len(left_images) != len(right_images):
            print(f"Erreur : nombre d'images différent ({len(left_images)} vs {len(right_images)})")
            return False
        
        print(f"Trouvé {len(left_images)} paires d'images")
        
        img_size = None
        successful_pairs = 0
        
        for i, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
            print(f"Traitement de la paire {i+1}/{len(left_images)}")
            
            # Trouve les coins dans l'image gauche
            ret_left, corners_left, img_left = self.find_corners(left_img_path)
            if not ret_left:
                print(f"  Échec : coins non trouvés dans l'image gauche {os.path.basename(left_img_path)}")
                continue
                
            # Trouve les coins dans l'image droite
            ret_right, corners_right, img_right = self.find_corners(right_img_path)
            if not ret_right:
                print(f"  Échec : coins non trouvés dans l'image droite {os.path.basename(right_img_path)}")
                continue
            
            # Si les deux ont réussi, ajoute les points
            self.objpoints.append(self.objp)
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            successful_pairs += 1
            
            # Récupère la taille de l'image
            if img_size is None:
                img_size = (img_left.shape[1], img_left.shape[0])
            
            # Optionnel : visualise les résultats
            if i < 3:  # Montre seulement les 3 premières paires
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                ax1.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
                ax1.set_title(f'Caméra 1 - Image {i+1}')
                ax1.axis('off')
                ax2.imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
                ax2.set_title(f'Caméra 2 - Image {i+1}')
                ax2.axis('off')
                plt.tight_layout()
                plt.savefig(f'calibration_pair_{i+1}.png')
                plt.close()
        
        print(f"\nPaires réussies : {successful_pairs}/{len(left_images)}")
        
        if successful_pairs < 10:
            print("Attention : moins de 10 paires réussies, la calibration pourrait être moins précise")
        
        # Calibration individuelle des caméras
        print("\nCalibration de la caméra gauche...")
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, img_size, None, None)
        
        print("Calibration de la caméra droite...")
        ret_right, self.camera_matrix_right, self.dist_coeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, img_size, None, None)
        
        # Calibration stéréo
        print("\nCalibration stéréo...")
        ret, self.camera_matrix_left, self.dist_coeffs_left, \
        self.camera_matrix_right, self.dist_coeffs_right, \
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            img_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        print(f"Erreur de reprojection : {ret:.3f}")
        print("\nCalibration terminée!")
        
        # Affiche les paramètres
        self.print_calibration_results()
        
        return True
    
    def print_calibration_results(self):
        """Affiche les résultats de calibration"""
        print("\n=== Résultats de calibration ===")
        print("\nMatrice de caméra gauche :")
        print(self.camera_matrix_left)
        print("\nCoefficients de distorsion gauche :")
        print(self.dist_coeffs_left.ravel())
        
        print("\nMatrice de caméra droite :")
        print(self.camera_matrix_right)
        print("\nCoefficients de distorsion droite :")
        print(self.dist_coeffs_right.ravel())
        
        print("\nRotation entre les caméras :")
        print(self.R)
        print("\nTranslation entre les caméras (en unités de square_size) :")
        print(self.T.ravel())
        
        # Calcule la baseline (distance entre les caméras)
        baseline = np.linalg.norm(self.T)
        print(f"\nDistance entre les caméras : {baseline:.1f} unités")
    
    def save_calibration(self, filename='stereo_calibration.npz'):
        """Sauvegarde les paramètres de calibration"""
        np.savez(filename,
                 camera_matrix_left=self.camera_matrix_left,
                 dist_coeffs_left=self.dist_coeffs_left,
                 camera_matrix_right=self.camera_matrix_right,
                 dist_coeffs_right=self.dist_coeffs_right,
                 R=self.R,
                 T=self.T,
                 E=self.E,
                 F=self.F)
        print(f"Calibration sauvegardée dans {filename}")
    
    def load_calibration(self, filename='stereo_calibration.npz'):
        """Charge les paramètres de calibration"""
        data = np.load(filename)
        self.camera_matrix_left = data['camera_matrix_left']
        self.dist_coeffs_left = data['dist_coeffs_left']
        self.camera_matrix_right = data['camera_matrix_right']
        self.dist_coeffs_right = data['dist_coeffs_right']
        self.R = data['R']
        self.T = data['T']
        self.E = data['E']
        self.F = data['F']
        print(f"Calibration chargée depuis {filename}")
    
    def rectify_images(self, left_image_path, right_image_path):
        """Rectifie une paire d'images pour la correspondance stéréo"""
        # Charge les images
        img_left = cv2.imread(left_image_path)
        img_right = cv2.imread(right_image_path)
        
        img_size = (img_left.shape[1], img_left.shape[0])
        
        # Calcule les matrices de rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            img_size, self.R, self.T, alpha=0)
        
        # Calcule les maps de rectification
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, R1, P1, img_size, cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, R2, P2, img_size, cv2.CV_32FC1)
        
        # Applique la rectification
        img_left_rect = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)
        
        return img_left_rect, img_right_rect, Q
    
    def compare_feature_detectors(self, left_image_path, right_image_path=None):
        """
        Compare différentes méthodes de détection de points d'intérêt
        sur une image (gauche par défaut, ou les deux si right_image_path est fourni)
        """
        # Charger l'image
        img = cv2.imread(left_image_path)
        if img is None:
            print(f"Impossible de charger l'image: {left_image_path}")
            return None
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Liste des détecteurs à comparer
        detectors = {}
        
        # Harris (cas spécial)
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        harris_threshold = 0.01 * dst.max()
        harris_points = np.argwhere(dst > harris_threshold)
        # Convertir en liste de KeyPoint pour la cohérence
        harris_keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 3) for x in harris_points]
        detectors["Harris Corner"] = harris_keypoints
        
        # Shi-Tomasi (nécessite conversion en KeyPoints)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        shi_tomasi_keypoints = []
        if corners is not None:
            corners = np.int32(corners)
            for corner in corners:
                x, y = corner.ravel()
                shi_tomasi_keypoints.append(cv2.KeyPoint(float(x), float(y), 3))
        detectors["Shi-Tomasi"] = shi_tomasi_keypoints
        
        # Autres détecteurs qui retournent directement des KeyPoints
        detectors["SIFT"] = cv2.SIFT_create().detect(gray, None)
        detectors["FAST"] = cv2.FastFeatureDetector_create(threshold=20).detect(gray, None)
        detectors["ORB"] = cv2.ORB_create(nfeatures=1000).detect(gray, None)
        detectors["BRISK"] = cv2.BRISK_create().detect(gray, None)
        detectors["AKAZE"] = cv2.AKAZE_create().detect(gray, None)
        
        # Statistiques et temps d'exécution
        performance = []
        detector_images = []
        
        # Figure pour afficher les résultats
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        # Pour chaque détecteur
        for i, (name, keypoints) in enumerate(detectors.items()):
            # Mesurer le temps d'exécution
            start_time = time.time()
            
            # Dessiner les keypoints
            if keypoints is not None:
                img_keypoints = cv2.drawKeypoints(
                    img, 
                    keypoints, 
                    None, 
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                execution_time = time.time() - start_time
                
                # Ajouter les statistiques
                performance.append([
                    name, 
                    len(keypoints), 
                    execution_time
                ])
                
                # Convertir pour matplotlib (BGR vers RGB)
                img_keypoints_rgb = cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB)
                detector_images.append(img_keypoints_rgb)
                
                # Afficher l'image
                if i < len(axes):
                    axes[i].imshow(img_keypoints_rgb)
                    axes[i].set_title(f'{name} ({len(keypoints)} points)')
                    axes[i].axis('off')
        
        # Afficher l'image originale
        if len(axes) > len(detectors):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[len(detectors)].imshow(img_rgb)
            axes[len(detectors)].set_title('Image originale')
            axes[len(detectors)].axis('off')
        
        # Masquer les axes inutilisés
        for i in range(len(detectors) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Comparaison des méthodes de détection de points d\'intérêt', fontsize=16)
        plt.subplots_adjust(top=0.95)
        plt.savefig('detectors_comparison.png')
        plt.show()
        
        # Afficher les statistiques sous forme de tableau
        print("\nStatistiques de performance :")
        print(tabulate(
            performance, 
            headers=["Détecteur", "Nombre de points", "Temps d'exécution (s)"],
            tablefmt="grid"
        ))
        
        # Si une image de droite est fournie, faire une comparaison des correspondances
        if right_image_path:
            print("\nComparaison des correspondances entre les images stéréo...")
            self.compare_stereo_matching(left_image_path, right_image_path, detectors)
        
        return performance, detector_images
    
    def compare_stereo_matching(self, left_image_path, right_image_path, detectors=None):
        """
        Compare les différents détecteurs pour la mise en correspondance stéréo
        après rectification des images
        """
        # Rectifie les images
        img_left_rect, img_right_rect, Q = self.rectify_images(left_image_path, right_image_path)
        
        # Convertit en niveaux de gris
        gray_left = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)
        
        # Si aucun détecteur n'est fourni, utiliser les plus courants
        if detectors is None:
            detectors = {
                "SIFT": cv2.SIFT_create().detect(gray_left, None),
                "ORB": cv2.ORB_create(nfeatures=1000).detect(gray_left, None),
                "BRISK": cv2.BRISK_create().detect(gray_left, None),
                "AKAZE": cv2.AKAZE_create().detect(gray_left, None)
            }
        
        # Statistiques pour chaque détecteur
        matching_stats = []
        
        # Figure pour afficher les résultats
        fig, axes = plt.subplots(len(detectors), 1, figsize=(20, 5*len(detectors)))
        
        # Assurer que axes est toujours une liste même avec un seul détecteur
        if len(detectors) == 1:
            axes = [axes]
        
        # Pour chaque détecteur
        for i, (name, kp_left) in enumerate(detectors.items()):
            # Skip Harris et Shi-Tomasi qui sont uniquement des détecteurs de coins
            if name in ["Harris Corner", "Shi-Tomasi"]:
                continue
                
            print(f"Traitement du détecteur {name}...")
            start_time = time.time()
            
            # Utiliser le même détecteur pour l'image droite
            if name == "SIFT":
                detector = cv2.SIFT_create()
            elif name == "ORB":
                detector = cv2.ORB_create(nfeatures=1000)
            elif name == "BRISK":
                detector = cv2.BRISK_create()
            elif name == "AKAZE":
                detector = cv2.AKAZE_create()
            elif name == "FAST":
                detector = cv2.FastFeatureDetector_create(threshold=20)
            else:
                continue  # Passer si le détecteur n'est pas reconnu
            
            # Détecter et calculer les descripteurs
            kp_left, desc_left = detector.detectAndCompute(gray_left, None)
            kp_right, desc_right = detector.detectAndCompute(gray_right, None)
            
            # Vérifier si des descripteurs ont été trouvés
            if desc_left is None or desc_right is None or len(desc_left) == 0 or len(desc_right) == 0:
                print(f"  Échec: pas de descripteurs trouvés avec {name}")
                continue
            
            # Matching selon le type de descripteur
            if name in ["SIFT"]:
                # Pour SIFT: utiliser BFMatcher avec knnMatch et ratio test de Lowe
                matcher = cv2.BFMatcher()
                raw_matches = matcher.knnMatch(desc_left, desc_right, k=2)
                
                # Appliquer le ratio test de Lowe
                good_matches = []
                try:
                    for m, n in raw_matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                except ValueError:
                    print(f"  Échec: problème avec knnMatch pour {name}")
                    continue
                
                matches = good_matches
            else:
                # Pour ORB, BRISK, etc: utiliser BFMatcher avec NORM_HAMMING et crossCheck
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(desc_left, desc_right)
            
            # Trier les matches par distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Limiter le nombre de matches pour la visualisation
            max_matches = min(50, len(matches))
            
            # Temps d'exécution total
            execution_time = time.time() - start_time
            
            # Ajouter les statistiques
            matching_stats.append([
                name,
                len(kp_left),
                len(kp_right),
                len(matches),
                execution_time
            ])
            
            # Dessiner les matches
            img_matches = cv2.drawMatches(
                img_left_rect, kp_left,
                img_right_rect, kp_right,
                matches[:max_matches], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Afficher l'image
            axes[i].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'{name}: {len(matches)} correspondances')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Comparaison des méthodes de mise en correspondance stéréo', fontsize=16)
        plt.subplots_adjust(top=0.95)
        plt.savefig('stereo_matching_comparison.png')
        plt.show()
        
        # Afficher les statistiques sous forme de tableau
        print("\nStatistiques de mise en correspondance stéréo :")
        print(tabulate(
            matching_stats,
            headers=["Détecteur", "Points gauche", "Points droite", "Correspondances", "Temps (s)"],
            tablefmt="grid"
        ))
        
        return matching_stats
    
    def find_correspondences(self, left_image_path, right_image_path, detector='ORB'):
        """Trouve les correspondances entre deux images avec le détecteur spécifié"""
        # Rectifie les images
        img_left_rect, img_right_rect, Q = self.rectify_images(left_image_path, right_image_path)
        
        # Convertit en niveaux de gris
        gray_left = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)
        
        # Crée le détecteur selon le choix
        if detector == 'ORB':
            det = cv2.ORB_create(nfeatures=1000)
        elif detector == 'SIFT':
            det = cv2.SIFT_create()
        elif detector == 'BRISK':
            det = cv2.BRISK_create()
        elif detector == 'AKAZE':
            det = cv2.AKAZE_create()
        else:
            print(f"Détecteur {detector} non reconnu, utilisation d'ORB par défaut")
            det = cv2.ORB_create(nfeatures=1000)
        
        # Détecte les points clés et calcule les descripteurs
        kp_left, desc_left = det.detectAndCompute(gray_left, None)
        kp_right, desc_right = det.detectAndCompute(gray_right, None)
        
        # Matching des descripteurs
        if detector == 'SIFT':
            matcher = cv2.BFMatcher()
            raw_matches = matcher.knnMatch(desc_left, desc_right, k=2)
            # Ratio test de Lowe
            good_matches = []
            for m, n in raw_matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            matches = good_matches
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(desc_left, desc_right)
        
        # Trie les matches par distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Visualise les correspondances
        img_matches = cv2.drawMatches(img_left_rect, kp_left, img_right_rect, kp_right,
                                    matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'Correspondances (top 50) - Détecteur: {detector}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'correspondances_{detector}.png')
        plt.show()
        
        # Affiche les lignes épipolaires pour vérifier la rectification
        self.show_epipolar_lines(img_left_rect, img_right_rect)
        
        return matches, kp_left, kp_right, Q
    
    def show_epipolar_lines(self, img_left, img_right):
        """Affiche les lignes épipolaires pour vérifier la rectification"""
        # Combine les images côte à côte
        combined = np.hstack((img_left, img_right))
        
        # Dessine des lignes horizontales
        h, w = img_left.shape[:2]
        for y in range(0, h, 50):
            cv2.line(combined, (0, y), (w*2, y), (0, 255, 0), 1)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.title('Lignes épipolaires (doivent être alignées après rectification)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('lignes_epipolaires.png')
        plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Crée l'objet de calibration
    stereo_cal = StereoCalibration(checkerboard_size=(9, 6), square_size=30)
    
    # Calibre les caméras
    print("Début de la calibration...")
    success = stereo_cal.calibrate_cameras(
        left_images_path="image/camera1/camera_1_image_*.jpg",
        right_images_path="image/camera2/camera_2_image_*.jpg"
    )
    
    if success:
        # Sauvegarde la calibration
        stereo_cal.save_calibration('stereo_calibration.npz')
        
        # Test sur une paire d'images
        print("\nTest de comparaison des détecteurs sur une paire d'images...")
        
        # Remplacez ces chemins par vos images de test
        test_left = "image/camera_1_image_calibration.jpg"
        test_right = "image/camera_2_image_calibration.jpg"
        
        if os.path.exists(test_left) and os.path.exists(test_right):
            # Charger une calibration existante si nécessaire
            # stereo_cal.load_calibration('stereo_calibration.npz')
            
            # Comparer les détecteurs de caractéristiques sur l'image gauche
            stereo_cal.compare_feature_detectors(test_left)
            
            # Comparer les détecteurs pour la mise en correspondance stéréo
            stereo_cal.compare_stereo_matching(test_left, test_right)
            
            # Utiliser le meilleur détecteur identifié pour trouver les correspondances
            print("\nRecherche de correspondances avec le détecteur recommandé...")
            matches, kp_left, kp_right, Q = stereo_cal.find_correspondences(test_left, test_right, detector='SIFT')
            print(f"Trouvé {len(matches)} correspondances avec SIFT")
        else:
            print("Images de test non trouvées")
    else:
        print("Échec de la calibration") 