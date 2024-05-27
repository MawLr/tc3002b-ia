# TC3002B-IA
José Manuel Medina - A01706212

**IMPORTANTE:** El dataset se encuntra en la siguiente carpeta de Google Drive: https://drive.google.com/drive/folders/1YqglgSbqy-tzIgb09LAQ0RlwOYUm0TtO?usp=sharing

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

Además de la separación, se aplicaron las siguientes técnicas de escalamiento: _rescale, rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip_
para así proporcionarle al modelo más datos con que entrenar sin alterar el rendimiento de las imágenes.

- rescale: Lo utilizamos para convertir los píxeles de las imágenes de rango `[0,255] a [0,1]`. Esto nos es de gran utilidad para normalizar las entradas y evitar posibles sesgos en los conteos de píxeles de cada imagen.
- rotation_range: Como lo dice su nombre, es un rango de rotación en ángulos, lo que significa que rota las entradas, ampliando los escenarios con los que el modelo se puede topar. En este caso se utilizó un rango de `40 grados` porque más de esto hace que se pueda alterar el patrón de las sonrisas en las cara y podría causar malas interpretaciones de las entradas.
- width_shift_range: Con esto desplazamos la imagen en el eje X, tanto a la izquierda como a la derecha dependiendo su valor. En nuestro caso el valor es de `0.15`, ya que el tamaño de la imagen es demasiado pequeño y un desplazamiento mayor hace que se pierda la expresión del cuadro.
- height_shift_range: Muy parecido a `width_shift_range` pero para el eje Y. El valor es también de `0.15` por la misma razón dada en el punto anterior.
- shear_range: Conocido en español como cizallamiento o mapeo de corte. Se podría decir que es un estiramiento de la imagen como si fuera jalada de dos puntos opuestos cruzados. En este caso el valor es de `0.2` nuevamente por las limitaciones de tamaño de las imagenes.
- zoom_range: Rango de enfoque, tiene un valor de `0.05` pare evitar perder facciones de las caras.
- horizontal_flip: Un espejeo de las imagenes, en nuestro caso está puesto como `True`.
