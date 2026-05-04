# Handwriting Recognition

This project applies machine learning techniques to analyze handwriting images and predict personality traits based on handwriting characteristics.

This project was developed as part of the **Methods and Techniques of Data Science** course at the **Faculty of Technical Sciences, University of Novi Sad**.

The goal is to classify handwritten samples according to the Big Five personality traits:

- Agreeableness
- Conscientiousness
- Extraversion
- Neuroticism
- Openness

## Project Overview

Handwriting can contain visual patterns related to writing style, spacing, shape, pressure, and structure. This project uses image-based handwriting data to extract relevant features and train classification models for personality trait prediction.

The workflow includes image processing, feature extraction, model training, and model evaluation.

---

## Repository Structure

```text
HandwritingRecognition/
│
├── Handwriting11.ipynb
├── extract.py
├── handwriting.py
├── knn.py
├── svm.py
└── README.md
```

---

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- pandas
- scikit-learn
- OpenCV
- K-Nearest Neighbors
- Support Vector Machine
- Neural Networks

---

## Models Used

The project explores and compares different machine learning models:

- K-Nearest Neighbors
- Support Vector Machine
- Neural Network-based approach

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/anaparovic/HandwritingRecognition.git
cd HandwritingRecognition
```

### 2. Install the required dependencies

```bash
pip install numpy pandas scikit-learn opencv-python jupyter
```

### 3. Run the notebook

```bash
jupyter notebook Handwriting11.ipynb
```

You can also run the Python scripts directly depending on the model you want to test:

```bash
python handwriting.py
python knn.py
python svm.py
```

---

## Results

The project compares different machine learning models for handwriting-based personality trait prediction. The results depend on the extracted handwriting features and the performance of each classification approach.

---

## Important Note

This project is intended for educational and research purposes. Personality prediction from handwriting is not scientifically reliable enough for sensitive or real-world decision-making and should be interpreted with caution.

---
