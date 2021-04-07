# Generación de datos sintéticos

Un modelo es entrenado con un dataset original para, a partir de este, generar nuevas muestras con las mismas características que las originales.

## Descripción

Los algoritmos para generar los datos sintéticos pertenecen al módulo [SDV](https://sdv.dev/SDV/index.html).
El módulo contiene modelos para datos relacionales, tabulares y series temporales; además de herramientas para evaluar los datos generados.

En la siguiente tabla se recogen todos los modelos existentes, resaltando en negrita los que han sido implementados (datos tabulares):

| Tipo de datos | Modelo | Abreviatura | Hiperparámetros |
| ------------- | ------ | ----------- | --------------- |
| **Tabulares** | **GaussianCopula Modelo** | GaussianCopula | |
| **Tabulares** | **CTGAN Model** | CTGAN | epochs, batch_size, log_frequency, embedding_dim, generator_dim , discriminator_dim, generator_lr, generator_decay, discriminator_lr, discriminator_decay, discriminator_steps, verbose, cuda |
| **Tabulares** | **CopulaGAN Model** | CopulaGAN | epochs, batch_size, log_frequency, embedding_dim, generator_dim , discriminator_dim, generator_lr, generator_decay, discriminator_lr, discriminator_decay, discriminator_steps, verbose |
| **Tabulares** | **TVAE Model** | TVAE | epochs, batch_size, log_frequency, embedding_dim, compress_dims, decompress_dims, l2scale, batch_size, loss_factor, cuda |
| Relacionales | HMA1 Class | | |
| Series temporales | PAR Model | | |

Los modelos marcados comparten los siguientes métodos:

* fit(X): Entrena el modelo con el DataFrame X.
* sample(n_muestras, conditions): Genera datos sintéticos en forma de DataFrame con *n_muestras* filas.
  
    La segunda entrada es opcional y permite establecer restricciones sobre los datos sintéticos, es decir, que las columnas deseadas adopten un valor determinado:
    `conditions = {'gender': 'M', experience_years': 0}`. Si esta entrada se da en forma de DataFrame, se generará una muestra por cada fila, siendo innecesario 
    definir *n_muestras*: `conditions = pd.DataFrame({'gender': ['M', 'M', 'M', 'F', 'F', 'F']})`
  
* get_distributions(): Devuelve las distribuciones estadísticas empleadas en cada columna.

## Implementación
[synthetic.py](https://github.com/josgarcam/synthetic_data/blob/main/synthetic.py)

Los modelos marcados se han recogido bajo la función **synthetic_generator(algorithm, parameters)**, cuyos argumentos de entrada son:

* algorithm: String con la abreviatura del modelo que se quiere emplear (tercera columna).
* parameters: Diccionario <u>opcional</u> con los parámetros del modelo:
    * primary_key: Se utiliza para indicar si alguna columna de la tabla se usa a modo de identificador y su valor debe ser único para cada entrada.
    * anonymize_fields: Para cuando el principal interés es preservar la privacidad de los datos y se quieren sustituir los originales por otros similares. 
      Se basa en el módulo [Faker](https://faker.readthedocs.io/en/master/index.html) y acepta tipos como *name, address, country...*
    * field_transformers: Para indicar el tipo de datos de cada columna y, por tanto, la transformación que se les realiza. Puede ser:
      *integer, float, categorical, categorical_fuzzy, one_hot_encoding, label_encoding, boolean y datetime*
    * field_distributions: Para indicar la distribución estadística a emplear en cada columna: *univariate, parametric, bounded, semi_bounded, parametric_bounded, 
      parametric_semi_bounded, gaussian, gamma, beta, student_t, gaussian_kde y truncated_gaussian*.
    * Alguno de los hiperparámetros del modelo recogidos en la tabla anterior.
      
    
    generator_params = {'primary_key': 'col_name',
                        'anonymize_fields': {'col_name': 'name'},
                        'field_transformers': {'address': 'label_encoding'},
                        'field_distributions': {experience_years': 'gamma'},
                        'epochs': 500,
                        'batch_size: 100,
                        'generator_dim: (256, 256, 256)
                       }

La función devuelve un objeto que integra el modelo. Este debe ser entrenado antes de ser usado para generar.

Nota: Si se quieren añadir más modelos tan solo es necesario incluirlos en el diccionario algorithms de la línea 10:

`algorithms = {'GaussianCopula': GaussianCopula, 'CTGAN': CTGAN, 'CopulaGAN': CopulaGAN, 'TVAE': TVAE}`

## Ejemplo de funcionamiento

En [main.py](https://github.com/josgarcam/synthetic_data/blob/main/main.py) se recoge un ejemplo de uso.
En este se genera un dataset ficticio con dos clústeres a modo de datos originales para entrenar un modelo de tipo GaussianCopula y, posteriormente, es empleado para generar 10 nuevas muestras.

![Figure_1](https://user-images.githubusercontent.com/80322524/113836185-f1084a00-978c-11eb-83c7-001571f20919.png)

## Evaluación

El módulo también incluye herramientas para evaluar los datos generados. En concreto, la función *evaluate* compara los datos originales y sintéticos mediante diferentes estadísticos.

`evaluate(synthetic_data, real_data, metrics=['CSTest', 'KSTest'], aggregate=False)`

La entrada *metrics* es opcional y si *aggregate* es true se devuelve un único valor calculado a partir de todos los estadísticos.

Para el ejemplo anterior:

| id | metric | name | score | min value | max value | goal |
| -- | ------ | ---- | ----- | --------- | --------- | ---- |
| 1 | LogisticDetection | LogisticRegression Detection | 1.000000 | 0.0 | 1.0 | MAXIMIZE |
| 2 | SVCDetection | SVC Detection | 0.313521 | 0.0 | 1.0 | MAXIMIZE |
| 11 | GMLogLikelihood | GaussianMixture Log Likelihood | -10.531854 | -inf | inf | MAXIMIZE |
| 13 | KSTest | Inverted Kolmogorov-Smirnov D statistic | 0.580000 | 0.0 | 1.0 | MAXIMIZE |
| 14 | KSTestExtended | Inverted Kolmogorov-Smirnov D statistic | 0.580000 | 0.0 | 1.0 | MAXIMIZE |
| 15 | ContinuousKLDivergence | Continuous Kullback–Leibler Divergence | 0.085780 | 0.0 | 1.0 | MAXIMIZE |

## Otras consideraciones

En la documentación también se recoge la posibilidad de guardar y cargar un modelo para usarlo posteriormente. También se detalla como establecer restricciones que engloban varias columnas.

