# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 06:21:38 2023

@author: Ana
"""

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def train_svm_model(data, targets):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.33, random_state=42)

    # Create an SVM model with the RBF kernel
    model = SVC(kernel='rbf')

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the accuracy of the model on the test set
    accuracy = model.score(X_test, y_test)

    return model, accuracy



