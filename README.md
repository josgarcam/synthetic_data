# Generación de datos sintéticos

Un modelo es entrenado con un dataset original para, a partir de este, genera nuevas muestras con las mismas características que las originales.

## Descripción

Hace uso de la técnica conocida como *Kernel Density Estimation* recogida en scikit-learn.
A grandes rasgos, dado un conjunto de muestras, estima su distribución de probabilidad y posteriormente la emula para generar nuevos datos.

Algunos parámetros requeridos para la creación del modelo son el bandwidth, el kernel y la métrica. 

## Enlaces de interés

[Kernel Density Estimation in Python Using Scikit-Learn](https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/)

[Generating Synthetic Data with Numpy and Scikit-Learn](https://stackabuse.com/generating-synthetic-data-with-numpy-and-scikit-learn/)

[KernelDensity()](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)

[GridSearchCV()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

## Resultados

![image](https://user-images.githubusercontent.com/80322524/114519225-79c92f00-9c40-11eb-93fe-d0e984d4a5fa.png)

![image](https://user-images.githubusercontent.com/80322524/114519271-864d8780-9c40-11eb-9100-17ccac56d70d.png)

![image](https://user-images.githubusercontent.com/80322524/114519326-96656700-9c40-11eb-991b-3c61ab5cf540.png)