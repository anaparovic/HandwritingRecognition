# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 06:35:18 2023

@author: Ana
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def treniraj_knn_model(podaci, ciljevi):

    # Standardizacija
    standardizator = StandardScaler()
    standardizovani_podaci = standardizator.fit_transform(podaci)

    # Podela podataka na trening i test skup
    X_train, X_test, y_train, y_test = train_test_split(podaci, ciljevi, test_size=0.33, random_state=42)

    # Kreiranje KNN modela sa brojem suseda (K) postavljenim na 3
    model = KNeighborsClassifier(n_neighbors=3)

    # Treniranje modela
    model.fit(X_train, y_train)

    # Procena tačnosti modela na test skupu
    tacnost = model.score(X_test, y_test)

    return model, tacnost

