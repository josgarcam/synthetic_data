from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Se generan dos nubes de puntos
blobs_params = dict(random_state=0, n_samples=200, n_features=2)
dataset = make_blobs(centers=[[4,  4], [-4, -4]], cluster_std=0.7, **blobs_params)[0]

# Se genera una zona de transición entre nubes
aux = np.random.uniform(-3.5, 3.5, (50, 1))
aux = np.c_[aux, aux + np.random.normal(size=aux.shape)]

# Se concatenan todas las muestras y se crea el dataset
dataset = np.concatenate((dataset, aux))
dataset = pd.DataFrame(data=dataset, columns=['x1', 'x2'])

# Modelo para la estimación de la función de densidad de probabilidad
model = KernelDensity()

# La función GridSearchCV permite encontrar cual es el ancho de banda que da lugar al mejor ajuste
bandwith_params = {'bandwidth': np.arange(0.01, 1, 0.05)}
grid_search = GridSearchCV(model, bandwith_params)

# Se ajusta el modelo con el dataset generado
grid_search.fit(dataset)

# Se extra el modelo que mejor se ha ajustado a los datos
kde = grid_search.best_estimator_

# Se generan los datos sintéticos
synthetic_dataset = kde.sample(40)

# Se representan los resultados
plt.figure()
plt.plot(dataset['x1'], dataset['x2'], 'o', alpha=0.8, label='Datos originales')
plt.plot(synthetic_dataset[:, 0], synthetic_dataset[:, 1], 'o', alpha=0.8, label='Datos sintéticos')
plt.legend()
plt.show()