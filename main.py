from synthetic import synthetic_generator
from sdv.evaluation import evaluate
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# warnings.filterwarnings('ignore')

# Generación del dataset de prueba
blobs_params = dict(random_state=0, n_samples=50, n_features=2)
dataset = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=0.5, **blobs_params)[0]
dataset = pd.DataFrame(data=dataset, columns=['col1', 'col2'])

# Creación del generador con los parámetros deseados
generator_params = {}
generator = synthetic_generator('GaussianCopula', generator_params)

# Entrenamiento del modelo
generator.fit(dataset)

# Generación de datos sintéticos
n_muestras_nuevas = 10
synthetic_dataset = generator.sample(n_muestras_nuevas)

# Evaluación del resultado
print(evaluate(synthetic_dataset, dataset, aggregate=False).to_string(), end='\n')

# Representación de los resultados
ax = dataset.plot.scatter('col1', 'col2', c='#00ff00', label='Original data')
synthetic_dataset.plot.scatter('col1', 'col2', c='#ff0000', ax=ax, label='Synthetic data')
ax.legend()
plt.show()

