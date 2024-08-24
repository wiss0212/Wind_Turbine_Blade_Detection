import cv2
import numpy as np

def draw_center_point(image_path, output_path, contours_output_path):
    # Charger l'image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Définir une région d'intérêt (ROI) plus stricte
    height, width = gray.shape
    roi = gray[int(height * 0.2):int(height * 0.9), int(width * 0.4):int(width * 0.6)]
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(roi, (9, 9), 0)
    cv2.imwrite("c:\\Users\\daouiaouissem\\Desktop\\Test_Technique\\blurred_image.jpg", blurred)
    
    # Appliquer le seuillage adaptatif pour obtenir une image binaire
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("c:\\Users\\daouiaouissem\\Desktop\\Test_Technique\\binary_image.jpg", binary)
    
    # Effectuer une fermeture (closing) pour éliminer les petits trous dans les objets
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("c:\\Users\\daouiaouissem\\Desktop\\Test_Technique\\closing_image.jpg", closing)
    
    # Appliquer la détection de contours Canny avec des seuils ajustés
    edges = cv2.Canny(closing, 100, 200)  # Réduction du seuil inférieur
    cv2.imwrite(contours_output_path, edges)
    
    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Aucun contour détecté.")
        return
    
    # Trouver le contour avec la plus grande aire parmi les contours détectés
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximation polygone pour s'assurer que le contour a une forme appropriée
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Calculer le centre du contour
    M = cv2.moments(approx)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Ajuster le centre détecté en fonction du ROI
    cX += int(width * 0.35)
    cY += int(height * 0.02)

    # maintenant on dessine un point au centre de l'objet 
    cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
    
    # Sauvegarder l'image résultante
    cv2.imwrite(output_path, image)

# utilisation de la fonction 
draw_center_point(
    "c:\\Users\\daouiaouissem\\Desktop\\Test_Technique\\DataPart1\\DataPart1\\P0000403.jpg", 
    "c:\\Users\\daouiaouissem\\Desktop\\Test_Technique\\image_result.jpg",
    "c:\\Users\\daouiaouissem\\Desktop\\Test_Technique\\contours_image.jpg"
)
