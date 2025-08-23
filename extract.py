# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 05:12:59 2023

@author: Ana
"""

import cv2
import numpy as np

#Gornja linija, vraca udaljenost od ivice
def extract_upper_margin(binary_image):
    # Find horizontal lines
    horizontal_lines = cv2.reduce(binary_image, 1, cv2.REDUCE_AVG).reshape(-1)
    threshold = 0.9 * np.max(horizontal_lines)  # Threshold for line detection
    
    lines = np.where(horizontal_lines > threshold)[0]
    
    # Extract the top boundary of the text
    top_boundary = lines[0]
    
    # Calculate the value of the upper margin
    upper_margin = top_boundary
    
    return upper_margin

#Pritisak olovke, vraca prosecnu vrednost pritiska olovke
def extract_pressure(binary_image):
    # Pronalaženje kontura
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtriranje kontura na osnovu površine
    min_contour_area = 100  # Minimalna površina konture koja će se uzeti u obzir
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Izračunavanje prosečne površine kontura kao mera pritiska olovke
    total_area = sum([cv2.contourArea(cnt) for cnt in filtered_contours])
    average_pressure = total_area / len(filtered_contours) if len(filtered_contours) > 0 else 0

    return average_pressure

#Ugao osnovne linije, vraca ugao na osnovu najvece konture na slici
def extract_baseline_angle(binary_image):
    # Pronalaženje kontura
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pronalaženje konture sa najvećom površinom
    largest_contour = max(contours, key=cv2.contourArea)

    # Pronalaženje najmanjeg pravougaonika koji obuhvata konturu
    rect = cv2.minAreaRect(largest_contour)

    # Izvlačenje ugla osnovne linije
    angle = rect[2]

    return angle

#Velicina slova, vraca prosecnu velicinu slova na osnovu najvece konture
def extract_font_size(binary_image):
    # Pronalaženje kontura
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Izračunavanje površine svake konture
    areas = [cv2.contourArea(contour) for contour in contours]

    # Pronalaženje konture sa najvećom površinom
    largest_contour_index = np.argmax(areas)
    largest_contour = contours[largest_contour_index]

    # Izračunavanje pravougaonika koji obuhvata konturu
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Izvlačenje visine slova
    font_size = h

    return font_size

#Razmak izmedju redova
def extract_line_spacing(binary_image):
    # Pronalaženje kontura
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sortiranje kontura po vertikalnoj poziciji
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Izračunavanje razmaka između centara susednih kontura
    spacings = []
    for i in range(1, len(contours)):
        _, y1, _, _ = cv2.boundingRect(contours[i-1])
        _, y2, _, _ = cv2.boundingRect(contours[i])
        spacing = y2 - (y1 + cv2.boundingRect(contours[i-1])[3])
        spacings.append(spacing)

    # Izračunavanje prosečnog razmaka između redova
    line_spacing = np.mean(spacings)

    return line_spacing

#Razmak izmedju slova
def extract_letter_spacing(binary_image):
    # Pronalaženje kontura
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sortiranje kontura po horizontalnoj poziciji
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Izračunavanje razmaka između desne ivice jedne konture i leve ivice sledeće konture
    spacings = []
    for i in range(1, len(contours)):
        x1, _, w1, _ = cv2.boundingRect(contours[i-1])
        x2, _, _, _ = cv2.boundingRect(contours[i])
        spacing = x2 - (x1 + w1)
        spacings.append(spacing)

    # Izračunavanje prosečnog razmaka između slova
    letter_spacing = np.mean(spacings)

    return letter_spacing

#Nagib slova
def extract_font_slant(binary_image):
    # Pronalaženje kontura
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Izračunavanje pravougaonika koji obuhvata sve konture
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))

    # Izvlačenje aspektnog odnosa pravougaonika
    aspect_ratio = float(w) / h

    # Izračunavanje nagiba slova
    font_slant = np.arctan(aspect_ratio) * (180 / np.pi)

    return font_slant
# Extract the upper margin values
#upper_margin = extract_upper_margin(binary_image)



def extract_features(image_list):
    upper_margins = []
    pressures = []
    baseline_angles = []
    font_sizes = []
    line_spacings = []
    letter_spacings = []
    font_slants = []

    for image in image_list:
        upper_margin = extract_upper_margin(image)
        upper_margins.append(upper_margin)
        
        pressure = extract_pressure(image)
        pressures.append(pressure)
        
        baseline_angle = extract_baseline_angle(image)
        baseline_angles.append(baseline_angle)
        
        font_size = extract_font_size(image)
        font_sizes.append(font_size)

        line_spacing = extract_line_spacing(image)
        line_spacings.append(line_spacing)

        letter_spacing = extract_letter_spacing(image)
        letter_spacings.append(letter_spacing)

        font_slant = extract_font_slant(image)
        font_slants.append(font_slant)

    # Konvertovanje listi u numpy nizove
    upper_margins= np.array(upper_margins)
    pressures = np.array(pressures)
    baseline_angles = np.array(baseline_angles)
    font_sizes = np.array(font_sizes)
    line_spacings = np.array(line_spacings)
    letter_spacings = np.array(letter_spacings)
    font_slants = np.array(font_slants)

    # Kreiranje matrice sa izvučenim parametrima
    features_matrix = np.column_stack((upper_margins, pressures, baseline_angles, font_sizes, line_spacings, letter_spacings, font_slants))

    return features_matrix


def transform_columns(matrix):
    # Transformacija prve kolone na 0 i 1
    min_value_1 = min(matrix[:, 0])
    max_value_1 = max(matrix[:, 0])
    transformed_column_1 = []
    for value in matrix[:, 0]:
        if value <= min_value_1 + (max_value_1 - min_value_1) / 2:
            transformed_column_1.append(0)
        else:
            transformed_column_1.append(1)
    
    # Transformacija druge kolone na 0, 1, 2
    min_value_2 = min(matrix[:, 1])
    max_value_2 = max(matrix[:, 1])
    transformed_column_2 = []
    for value in matrix[:, 1]:
        if value <= min_value_2 + (max_value_2 - min_value_2) / 3:
            transformed_column_2.append(0)
        elif value <= min_value_2 + 2 * (max_value_2 - min_value_2) / 3:
            transformed_column_2.append(1)
        else:
            transformed_column_2.append(2)

    # Transformacija treće kolone na 0, 1, 2
    min_value_3 = min(matrix[:, 2])
    max_value_3 = max(matrix[:, 2])
    transformed_column_3 = []
    for value in matrix[:, 2]:
        if value <= min_value_3 + (max_value_3 - min_value_3) / 3:
            transformed_column_3.append(0)
        elif value <= min_value_3 + 2 * (max_value_3 - min_value_3) / 3:
            transformed_column_3.append(1)
        else:
            transformed_column_3.append(2)

    # Transformacija četvrte kolone na 0, 1, 2
    min_value_4 = min(matrix[:, 3])
    max_value_4 = max(matrix[:, 3])
    transformed_column_4 = []
    for value in matrix[:, 3]:
        if value <= min_value_4 + (max_value_4 - min_value_4) / 3:
            transformed_column_4.append(0)
        elif value <= min_value_4 + 2 * (max_value_4 - min_value_4) / 3:
            transformed_column_4.append(1)
        else:
            transformed_column_4.append(2)

    # Transformacija pete kolone na 0, 1, 2
    min_value_5 = min(matrix[:, 4])
    max_value_5 = max(matrix[:, 4])
    transformed_column_5 = []
    for value in matrix[:, 4]:
        if value <= min_value_5 + (max_value_5 - min_value_5) / 3:
            transformed_column_5.append(0)
        elif value <= min_value_5 + 2 * (max_value_5 - min_value_5) / 3:
            transformed_column_5.append(1)
        else:
            transformed_column_5.append(2)

    # Transformacija šeste kolone na 0, 1, 2
    min_value_6 = min(matrix[:, 5])
    max_value_6 = max(matrix[:, 5])
    transformed_column_6 = []
    for value in matrix[:, 5]:
        if value <= min_value_6 + (max_value_6 - min_value_6) / 3:
            transformed_column_6.append(0)
        elif value <= min_value_6 + 2 * (max_value_6 - min_value_6) / 3:
            transformed_column_6.append(1)
        else:
            transformed_column_6.append(2)

    # Transformacija sedme kolone na 0, 1, 2, 3, 4, 5, 6
    min_value_7 = min(matrix[:, 6])
    max_value_7 = max(matrix[:, 6])
    transformed_column_7 = []
    for value in matrix[:, 6]:
        interval_length = (max_value_7 - min_value_7) / 6
        transformed_value = int((value - min_value_7) // interval_length)
        transformed_column_7.append(transformed_value)

    # Ažuriranje prvih sedam kolona matrice sa transformisanim vrednostima
    matrix[:, 0] = transformed_column_1
    matrix[:, 1] = transformed_column_2
    matrix[:, 2] = transformed_column_3
    matrix[:, 3] = transformed_column_4
    matrix[:, 4] = transformed_column_5
    matrix[:, 5] = transformed_column_6
    matrix[:, 6] = transformed_column_7

    return matrix


