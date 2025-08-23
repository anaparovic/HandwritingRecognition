# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 01:00:13 2023

@author: Ana
"""

import extract

import cv2
import os
import numpy as np
import svm
import knn
import matplotlib.pyplot as plt


# Putanja do slika za treinranje
folder_path = "C:/Users/Antonela/Desktop/Fakuktet/3/MITNOP/Handwriting11/dataset/training_set"

# Putanja do slika bez suma i sa binarizacijom
filtered_folder_path = "C:/Users/Antonela/Desktop/Fakuktet/3/MITNOP/Handwriting11/dataset/filtered_training_set"

#Putanja do slika nakon dilatacije, konture i afine transformacije
dialation_folder_path = "C:/Users/Antonela/Desktop/Fakuktet/3/MITNOP/Handwriting11/dataset/final_training_set"

# Gde cuvamo slike i oznake osobina
images = []
labels = []

# Gde cuvamo slike posle uklanjanja suma
filtered_images = []

#Finalno obradjene slike
final_images = []

# Prolazimo kroz trening folder
for folder in os.listdir(folder_path):
    folder_full_path = os.path.join(folder_path, folder)
    filtered_folder_full_path = os.path.join(filtered_folder_path, folder)
    dialation_folder_full_path = os.path.join(dialation_folder_path, folder)


    # Prolazak kroz sve slike
    for image_name in os.listdir(folder_full_path):
        image_path = os.path.join(folder_full_path, image_name)

        
        # Ucitavanje slike
        image = cv2.imread(image_path)
        
        # Eliminisanje suma
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Konverzija u sivu skaliranu verziju
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        
        # Binarizacija koristeci globalni inverzni prag
        _, binary_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)
        
        # Ispravljanje linija rukopisa

        # Dilatacija
        kernel = np.ones((5, 100), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

        # Pronalaženje kontura
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sortiranje kontura po površini
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Pronalaženje najveće konture
        largest_contour = contours[0]

        # Izdvajanje pravougaonika koji obuhvata najveću konturu
        x, y, w, h = cv2.boundingRect(largest_contour)
        

        # Ispravljanje slike koristeći afine transformacije
        points1 = np.float32([[x, y], [x + w, y], [x, y + h]])
        points2 = np.float32([[0, 0], [w, 0], [0, h]])
        transform_matrix = cv2.getAffineTransform(points1, points2)
        corrected_image = cv2.warpAffine(binary_image, transform_matrix, (w, h))
        
        
        # Sacuvaj
        filtered_image_path = os.path.join(filtered_folder_full_path, image_name.replace(os.sep, "/"))
        filtered_image_path = filtered_image_path.replace("/", "\\")  # Zamenjujemo sve \ sa /
        cv2.imwrite(filtered_image_path, binary_image)
        
        dialation_image_path = os.path.join(dialation_folder_full_path, image_name.replace(os.sep, "/"))
        dialation_image_path = dialation_image_path.replace("/", "\\")  # Zamenjujemo sve \ sa /
        cv2.imwrite(dialation_image_path, corrected_image)
        
        # Dodaj sve u listu
        images.append(filtered_image)
        labels.append(folder)
        filtered_images.append(binary_image)
        final_images.append(corrected_image)

    
# Koliko slika je ucitano
print("Total images loaded:", len(images))


# Koliko slika liseno suma
print("Total filtered images:", len(filtered_images))

#Ekstraktovanje parametara
features_matrix = extract.extract_features(final_images)
#print(features_matrix)
transformed_matrix = extract.transform_columns(features_matrix)

model_svm, accuracy_svm = svm.train_svm_model(transformed_matrix, labels)
print("Accuracy of SVM is: ", accuracy_svm)

model_knn, accuracy_knn = knn.treniraj_knn_model(features_matrix, labels)
print("Accuracy of KNN is: ", accuracy_knn)


#Predikcija

in_images = []

folder_path = "C:/Users/Antonela/Desktop/Fakuktet/3/MITNOP/Handwriting11/dataset/informacioni_inzenjering"
for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)

    # Provera da li je datoteka slika
    if os.path.isfile(image_path):
        # Učitavanje slike
        image = cv2.imread(image_path)
        
        # Eliminisanje šuma
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Konverzija u sivu skaliranu verziju
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        
        # Binarizacija koristeći globalni inverzni prag
        _, binary_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)
        
        # Ispravljanje linija rukopisa
        
        # Dilatacija
        kernel = np.ones((5, 100), np.uint8)
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

        # Pronalaženje kontura
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sortiranje kontura po površini
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Pronalaženje najveće konture
        largest_contour = contours[0]

        # Izdvajanje pravougaonika koji obuhvata najveću konturu
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ispravljanje slike koristeći afine transformacije
        points1 = np.float32([[x, y], [x + w, y], [x, y + h]])
        points2 = np.float32([[0, 0], [w, 0], [0, h]])
        transform_matrix = cv2.getAffineTransform(points1, points2)
        corrected_image = cv2.warpAffine(binary_image, transform_matrix, (w, h))
        
        # Sačuvaj slike
        filtered_image_path = os.path.join(image_path, image_name)
        cv2.imwrite(filtered_image_path, binary_image)
      
        # Dodaj slike u listu
        images.append(filtered_image)
        labels.append("informacioni_inzenjering")
        filtered_images.append(binary_image)
        in_images.append(corrected_image)
        
        

# Primena modela na preprocesirane slike
in_ekstraktovano = extract.extract_features(in_images)
in_transformisano = extract.transform_columns(in_ekstraktovano)
rezultati = model_svm.predict(in_transformisano)

print("\nPredikcije smera prema SVM algoritmu:/n")
# Prikazivanje rezultata
for i, klasa in enumerate(rezultati):
    print("Slika {} pripada klasi {}".format(i, klasa))
    
    
# Primena modela na preprocesirane slike
in_ekstraktovano = extract.extract_features(in_images)
in_transformisano = extract.transform_columns(in_ekstraktovano)
rezultati = model_knn.predict(in_transformisano)
print("\nPredikcije smera prema KNN algoritmu:/n")

# Prikazivanje rezultata
for i, klasa in enumerate(rezultati):
    print("Slika {} pripada klasi {}".format(i, klasa))
    
    





