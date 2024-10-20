import cv2
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
import pandas as pd
import math

taille_reelle_pixel = 3.6e-3

# Étape 1 : Charger l'image
image = cv2.imread('C:/Users/tomde/Desktop/Cours/Autonme 2024/Optique/Laboratoire 2/Res/res_1m_min.png')

# Obtenir la hauteur de l'image
image_height = image.shape[0]
half_height = image_height // 2  # Moitié de l'image

# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un flou gaussien pour réduire le bruit
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Appliquer un seuillage adaptatif binaire inversé avec un blockSize petit
thresh = cv2.adaptiveThreshold(
    gray_blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=11,  # BlockSize petit
    C=2
)

# Opérations morphologiques pour améliorer les formes
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# Fermeture pour combler les petits trous
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
# Ouverture pour supprimer les petits objets
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Détection des contours
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Créer une copie de l'image pour le dessin
image_contours = image.copy()

# Liste pour stocker les longueurs des rectangles
rectangle_lengths = []

# Compteur de rectangles
rectangle_counter = 1

# Liste des longueurs à supprimer
longueurs_a_supprimer = [37, 69, 78, 79, 87, 103, 104, 105, 106, 107, 111, 112, 173]

# Condition spécifique : longueur = 87 et largeur > 12
longueur_specific = 87
largeur_minimale = 12

for contour in contours:
    # Calculer l'aire et le périmètre du contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Filtrer les contours de petite taille
    if area > 600:  # Ajuster ce seuil en fonction de votre image
        # Approximation du contour
        epsilon = 0.02 * perimeter  # Garder epsilon petit pour une approximation précise
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Vérifier si c'est un quadrilatère
        if len(approx) == 4:
            # Vérifier si le contour est convexe
            if cv2.isContourConvex(approx):
                # Utiliser le rectangle englobant pour calculer la longueur et la largeur
                x, y, w, h = cv2.boundingRect(approx)
                length = max(w, h)
                width = min(w, h)

                # Vérifier si la position verticale est au-dessus de la moitié de l'image
                if y + h // 2 < half_height:
                    # Si le centre du rectangle est au-dessus de la moitié de l'image
                    length_to_use = width  # On utilise la largeur
                else:
                    length_to_use = length  # Sinon, on garde la longueur

                # Vérifier si la longueur fait partie des longueurs à supprimer
                if int(length_to_use) in longueurs_a_supprimer:
                    continue  # Ne pas dessiner ni afficher cette longueur
                    
                if abs(width - length) < 20:
                    continue 

                rectangle_lengths.append(length_to_use)

                # Dessiner le contour
                cv2.drawContours(image_contours, [approx], -1, (0, 255, 0), 2)

                # Afficher le numéro du rectangle et sa longueur ou largeur sur l'image
                cX = x + w // 2
                cY = y + h // 2
                text = f"Dimension : {int(length_to_use)}"
                cv2.putText(
                    image_contours,
                    text,
                    (cX - 70, cY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )
                rectangle_counter += 1
        else:
            # Gestion des contours non quadrilatéraux
            rect = cv2.minAreaRect(contour)
            (center), (width, height), angle = rect
            length = max(width, height)
            width = min(width, height)

            # Vérifier si la position verticale est au-dessus de la moitié de l'image
            if center[1] < half_height:
                # Si le centre du rectangle est au-dessus de la moitié de l'image
                length_to_use = width  # On utilise la largeur
            else:
                length_to_use = length  # Sinon, on garde la longueur

            # Vérifier si la longueur fait partie des longueurs à supprimer
            if int(length_to_use) in longueurs_a_supprimer:
                continue  # Ne pas dessiner ni afficher cette longueur

            rectangle_lengths.append(length_to_use)

            # Obtenir les points du rectangle
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Dessiner le rectangle
            cv2.drawContours(image_contours, [box], 0, (0, 0, 255), 2)

            # Afficher le numéro du rectangle et sa longueur ou largeur sur l'image
            cX, cY = np.intp(center)
            text = f"Dimension : {int(length_to_use)}"
            cv2.putText(
                image_contours,
                text,
                (int(cX) - 70, int(cY)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
            rectangle_counter += 1


# Afficher l'image avec les rectangles détectés
image_rgb = cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# Calcul de la moyenne et de l'écart-type pour chaque groupe de rectangles similaires

# Trier les longueurs détectées
rectangle_lengths.sort()

# Grouper les rectangles similaires (différence maximale de 10)
grouped_lengths = []
current_group = [rectangle_lengths[0]]

for i in range(1, len(rectangle_lengths)):
    if abs(rectangle_lengths[i] - rectangle_lengths[i - 1]) <= 10:
        current_group.append(rectangle_lengths[i])
    else:
        grouped_lengths.append(current_group)
        current_group = [rectangle_lengths[i]]

# Ajouter le dernier groupe
if current_group:
    grouped_lengths.append(current_group)

# Associer chaque groupe à une taille en mm (1mm, 2mm, 10mm, 25mm, 50mm)
group_to_size = {1: '1', 2: '2', 3: '10', 4: '25', 5: '50'}

# Création des colonnes pour la taille du groupe, la moyenne et l'écart-type
data = []

# Calcul de la moyenne et de l'écart-type pour chaque groupe
for idx, group in enumerate(grouped_lengths, 1):
    if len(group) > 1:  # Calculer uniquement si le groupe contient plus d'un rectangle
        group_std = stdev(group)
    else:
        group_std = 0
    group_mean = mean(group)
    size = group_to_size.get(idx, "Inconnu")
    data.append([size, group_mean, group_std])

# Création du tableau avec pandas
df = pd.DataFrame(data, columns=['Taille (mm)', 'Moyenne (pixels)', 'Écart-type (pixels)'])

print(df)

# Calcul de la résolution (taille en mm / moyenne en pixels)
df['Taille (mm)'] = df['Taille (mm)'].astype(float)  # Convertir la taille en numérique
df['Résolution (mm/pixel)'] = df['Taille (mm)'] / df['Moyenne (pixels)']

# Calcul de la moyenne et de l'écart-type des résolutions
resolution_mean = df['Résolution (mm/pixel)'].mean()
resolution_std = df['Résolution (mm/pixel)'].std()

if math.isnan(resolution_std):
    resolution_std = 0

# Afficher le tableau des statistiques de résolution
print(f"\nMoyenne de la résolution : {resolution_mean:.6f} mm/pixel")
print(f"Écart-type de la résolution : {resolution_std:.6f} mm/pixel")

# Calcul du grossissement
df['Taille (mm)'] = df['Taille (mm)'].astype(float)  # Convertir la taille en numérique
df['Grossissement (mm/pixel)'] = (df['Moyenne (pixels)'] * taille_reelle_pixel)/df['Taille (mm)']

# Calcul de la moyenne et de l'écart-type du grossissement
grossissement_mean = df['Grossissement (mm/pixel)'].mean()
grossissement_std = df['Grossissement (mm/pixel)'].std()

if math.isnan(grossissement_std):
    grossissement_std = 0

# Afficher le tableau des statistiques du grossissement
print(f"\nMoyenne du grossissement : {grossissement_mean:.6f}")
print(f"Écart-type du grossissement : {grossissement_std:.6f}")

# Sauvegarder l'image modifiée
output_image_path = 'C:/Users/tomde/Desktop/Cours/Autonme 2024/Optique/Laboratoire 2/Res/rectangles_detectes_min.png'
cv2.imwrite(output_image_path, image_contours)

# Sauvegarder le tableau en LaTeX
latex_table = df.to_latex(index=False)

# Écrire le tableau LaTeX dans un fichier .tex
output_latex_path = 'C:/Users/tomde/Desktop/Cours/Autonme 2024/Optique/Laboratoire 2/Res/tableau_longueurs_min.tex'
with open(output_latex_path, 'w') as f:
    f.write(latex_table)

#print(f"Image sauvegardée : {output_image_path}")
#print(f"Tableau LaTeX sauvegardé : {output_latex_path}")
