from sdv.tabular import GaussianCopula
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN
from sdv.tabular import TVAE


def synthetic_generator(algorithm, parameters):
    # Recibe una cadena con el nombre del algoritmo a emplear, y un diccionario con los parámetros de este.

    algorithms = {'GaussianCopula': GaussianCopula, 'CTGAN': CTGAN, 'CopulaGAN': CopulaGAN, 'TVAE': TVAE}
    model = algorithms[algorithm](**parameters)
    return model
