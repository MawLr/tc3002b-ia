# TC3002B-IA
José Manuel Medina - A01706212

**IMPORTANTE:** El dataset se encuntra en la siguiente carpeta de Google Drive: https://drive.google.com/drive/folders/1YqglgSbqy-tzIgb09LAQ0RlwOYUm0TtO?usp=sharing

**IMPORTANTE X2:** El _notebook_ con todos los modelos se encuentra en el archivo `modelo_base_smile_final.ipynb`.

# Modelo de detección de sonrisas en caras humanas
Utilizando inteligencia artificial, se busca crear un modelo que pueda reconocer si una persona está
sonriendo o no.

Hay dos clases:
- smile
- non_smile

## Descripción del Dataset
El dataset de sonrizas se obtuvo de Kaggle, siendo este el [Smiling or Not | Face Data](https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data) por el autor Chazzer.
Dentro del dataset se encuentran las siguientes carpetas:
- `smile` - 600 imágenes de 64x64 px
- `non_smile` - 603 imágenes de 64x64 px
- `test` - ≈12,000 ±100 imágenes, _sin clasificar_ de 64x64 px

Por cuestiones de practicidad únicamente se tomaron los archivos de las carpetas `smile` y `non_smile`.

## Preprocesado de los datos
Una vez seleccionado el dataset se llevó a cabo la separación de los datos de entrenamiento y de prueba.
Dentro de la carpeta `model` se crearon las carpetas `train` y `test` con las respectivas dos clases originales del
dataset: `smile` y `non_smile`. 

Se hizo una división de 80-20, es decir, 80% de los datos se destinaron a `train` y el
20% restante a `test`, tal como se recomienda en la infografía _"Building the Machine Learning Model"_ de Nantasenamat, C. (2020).

Dado a que las imágenes se encuentran acomodadas por orden alfabético (las imágenes tienen el nombre de la persona), se tomaron los primeros 120 elementos de `smile` y `non_smile`
para hacer la separación inicial de `train` y `test`, dando un total de 240, que es ≈20% del total de elementos (1,203).

Con los datos de `test` se hizo una separación más, la mitad de las imagenes dentro de esta carpeta se destinó a `validation`, quedando así una separación de 80% `train`, 10% `test` y 10% `validation`.

Además de la separación, se aplicaron las siguientes técnicas de escalamiento: _rescale, rotation_range, width_shift_range, shear_range, horizontal_flip_
para así proporcionarle al modelo más datos con que entrenar sin alterar el rendimiento de las imágenes.

- rescale: Lo utilizamos para convertir los píxeles de las imágenes de rango `[0,255] a [0,1]`. Esto nos es de gran utilidad para normalizar las entradas y evitar posibles sesgos en los conteos de píxeles de cada imagen.
- rotation_range: Como lo dice su nombre, es un rango de rotación en ángulos, lo que significa que rota las entradas, ampliando los escenarios con los que el modelo se puede topar. En este caso se utilizó un rango de `40 grados` porque más de esto hace que se pueda alterar el patrón de las sonrisas en las cara y podría causar malas interpretaciones de las entradas.
- width_shift_range: Con esto desplazamos la imagen en el eje X, tanto a la izquierda como a la derecha dependiendo su valor. En nuestro caso el valor es de `0.1`, ya que el tamaño de la imagen es demasiado pequeño y un desplazamiento mayor hace que se pierda la expresión del cuadro.
- shear_range: Conocido en español como cizallamiento o mapeo de corte. Se podría decir que es un estiramiento de la imagen como si fuera jalada de dos puntos opuestos cruzados. En este caso el valor es de `0.2` nuevamente por las limitaciones de tamaño de las imagenes.
- horizontal_flip: Un espejeo de las imagenes, en nuestro caso está puesto como `True`.

## Modelo Inicial
El modelo inicial es un modelo Secuencial Binario y consta de siete capas: 
- 2 capas Conv2D utilizando `ReLu` como algoritmo de activación
- 2 capas MaxPool2D metidas entre cada Conv2D
- 1 capa de Flatten
- 1 capa Densa de 64 nodos con activación `ReLu`
- 1 capa Densa de 1 nodo (ya que es un modelo binario) con activación `sigmoid`

Se optó por usar una arquitectura similar a una CNN con base en el artículo de Guo, X. & Polaina, L. F. (2018) Igualmente, el uso de una capa Conv2D seguida de una MaxPool2D con los algoritmos de activación `ReLu` fue tomado del artículo de Malallah, F. et al. (2020). La última capa densa utiliza el algoritmo de activación `sigmoid` dado a que éste nos retorna valores entre 1 y 0, por lo que es de gran utilidad en clasificación binaria.

### Métricas y consideraciones
Para este modelo inicial utilizamos:
- Binary Cross-Entropy: Utilizada como función de _Loss_. Mide la disimilitud entre los _labels_ actuales y las probabilidades predichas en la clase positiva (1 pos. 0 neg.) y penaliza las predicciones que tienen confiabilidad pero son incorrectas. El concepto puede aplicar para más de dos clases (_Categorical_) pero en nuestro caso es ideal el utilizar la función binaria.
- Accuracy: Método para medir el desempeño del modelo. Cuenta las predicciones donde el valor predicho es igual al verdadero valor.
- Loss: Suma de errores cometidos por cada muestra en el set de entrenamiento. Toma en cuenta las probabilidades o incertidumbre de una predicción basada en qué tanto varía la predicción con el valor real.
- RMSProp: Método de optimización. _Root Mean Square Propagation_ por sus siglas en inglés. Se decidió utilizar este optimizador con base en la publicación de López-Sánchez, M. et al. (2021) que compara el desempeño de SGD, RMSProp y Adam. Cabe mencionar que RMSProp fue el que tuvo resultados intermedios y Adam fue el que tuvo mejor desempeño, pero para este primer modelo utilizamos RMSProp como punto de partida.

### Evaluación inicial
El modelo inicial nos arroja un _Accuracy_ de 92%% y un _F1 Score_ de 91%. A primera vista, y con base en lo reportado por Guo, X. & Polaina, L. F. (2018), nuestro modelo tiene un porcentaje de _Accuracy_ comparable con un modelo que utiliza _HOG (Labelled + unlabelled)_ como _Features_ y un _Classifier_ SVM, ya que este es de ±92%.

Considerando que el _validation loss_ (0.2213) es similar al _train loss_ (0.2935) y que tanto _train accuracy_ (88.99%), _validation accuracy_ (90%) y _test accuracy_ (92.25%) son medianamente similares, podemos decir que el modelo parece haber sido entrenado de manera correcta. Cabe mencionar que quizá implementando algunas de las mejoras propuestas en los artículos citados el porcentaje de _accuracy_ podría subir, pero como primera iteración parece ir por buen camino.

#### Propuestas para el siguiente modelo
- Cambiar RMSProp por Adam

## Segundo modelo
El segundo es un modelo Secuencial Binario y consta de siete capas: 
- 2 capas Conv2D utilizando `ReLu` como algoritmo de activación
- 2 capas MaxPool2D metidas entre cada Conv2D
- 1 capa de Flatten
- 1 capa Densa de 64 nodos con activación `ReLu`
- 1 capa Densa de 1 nodo (ya que es un modelo binario) con activación `sigmoid`

La estructura es la misma a la del modelo inicial pero por propósitos de experimentación se reemplazó el _optimizer_ de `RMSprop` a `Adam`. Igualmente, en vez de tener 10 filtros en las dos capas Conv2D, empezamos 
con 10 y la segunda capa continúa con 15. La capa densa pasó de 64 -> 128 perceptrones.

### Evaluación inicial
El modelo arroja resultados similares a los del primer modelo, con un _Accuracy_ de 90% y un _F1 Score_ de 89%. A primera vista no parece haber una mejoría significativa con los pequeños cambios hechos al modelo.

El _validation loss_ es de 0.3588 y el _train loss_ es de 0.2353. 
_Train accuracy_ es de 90%, _validation accuracy_ es de 90% y _test accuracy_ es de 90%.

#### Propuestas para el siguiente modelo
- El modelo de Guo, X. & Polaina, L. F. (2018) propone utilizar una capa VGG-16, sería interesante añadirla al modelo y observar cómo se comporta

## Tercer modelo
El tercer modelo es igualmente uno Secuencial Binario, consta de la siguientes capas:
- 16 capas base VGG16 utilizando `imagenet` como _weights_
- 2 capas Conv2D utilizando `ReLu` como algoritmo de activación
- 2 capas MaxPool2D metidas entre cada Conv2D
- 1 capa de Flatten
- 1 capa Densa de 64 nodos con activación `ReLu`
- 1 capa Densa de 1 nodo (ya que es un modelo binario) con activación `sigmoid`

Al igual que el segundo modelo, los filtros de las capas Conv2D van de 10 -> 15.
El _optimizer_ es nuevamente `Adam`.

### Evaluación inicial
Este modelo tuvo una ligera pérdida de _Accuracy_, con un 87.5% y un _F1 Score_ de 87%. A pesar de que empeoró el resultado, si analizamos _train loss_ (0.2824) contra _validation loss_ (0.4384)
podemos inferir que aún se puede ajustar ligeramente el modelo para obtener mejores resultados, ya que no parece tener ni _overfitting_ ni _underfitting_.

Sería interesante si con suficiente poder de cómputo para hacer el entrenamiento de manera rápida se entrenaran las capas del VGG16, ya que en este modelo _trainable_ se estableció como _false_.

## Evaluación final de los modelos

| Modelo                    | Accuracy | Precision | Recall | F1 Score |
|---------------------------|----------|-----------|--------|----------|
| V1(CNN + RMSProp)         | 92%      | 85%       | 100%   | 91%      |
| V2(CNN + Adam)            | 90%      | 85%       | 94%    | 89%      |
| V3(CNN + VGG16 + Adam)    | 87.5%    | 85%       | 89%    | 87%      |

En conclusión, el modelo inicial fue el que obtuvo los mejores resultados, quizá porque es el que menos parámetros utilizó (puesto que tenía menos filtros que los demás en las capas Conv2D).
La complejidad de agregar 16 capas adicionales puede que haya influído en el notable cambio de _Accuracy_ en el tercer modelo.

Es importante mencionar que el _Recall_ del 100% en el modelo V1 significa que no tuvo falsos negativos, por lo que podemos decir que tiene una alta sensibilidad para detectar casos positivos.
También cabe destacar que aunque López-Sánchez, M. et al. (2021) proponen que el mejor optimizador para CNNs es `Adam`, hay ciertamente ocasiones donde considerar alternativas como `RMSProp` pueden
influir de manera positiva al modelo, como lo fue en el caso del V1.

## Referencias:
- Nantasenamat, C. (2020). _Building the Machine Learning Model_. Recuperado de: https://towardsdatascience.com/how-to-build-a-machine-learning-model-439ab8fb3fb1

- Guo, X. & Polaina, L. F. (2018). _Smile Detection in the Wild Based on Transfer Learning_. http://dx.doi.org/10.1109/FG.2018.00107

- Malallah, F., Al-Jubouri, A., Sabaawi, A., Shareef, B., Saeed, M. & Yasen, K. (2020). _Smiling and Non-smiling Emotion Recognition Based on Lower-half Face using Deep-Learning as Convolutional Neural Network_. http://dx.doi.org/10.4108/eai.28-6-2020.2298175

- López-Sánchez, M., Hernández-Torruco, J., Hernández-Ocaña, B. & Chávez-Bosquez, O. (2021, Febrero 10). Comparative Study of Optimizers in the Training of a Convolutional Neural Network in a Binary Recognition Model. _Research in Computing Science, 150(4)_, 73-82. https://rcs.cic.ipn.mx/2021_150_4/Comparative%20Study%20of%20Optimizers%20in%20the%20Training%20of%20a%20Convolutional%20Neural%20Network.pdf   
