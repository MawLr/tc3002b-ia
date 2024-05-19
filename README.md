# tc3002b-ia
José Manuel Medina - A01706212

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
20% restante a `test`, tal como se recomienda en la infografía _"Building the Machine Learning Model"_ de Nantasenanat, C. (2020).

Dado a que las imágenes se encuentran acomodadas por orden alfabético (las imágenes tienen el nombre de la persona), se tomaron los primeros 120 elementos de `smile` y `non_smile`
para hacer la separación inicial de `train` y `test`, dando un total de 240, que es ≈20% del total de elementos (1,203).

Además de la separación, se aplicaron técnicas de escalamiento _(rescale, rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip)_
para así proporcionarle al modelo más datos con que entrenar sin alterar el rendimiento de las imágenes.

**IMPORTANTE:** El dataset se encuntra en la siguiente carpeta de Google Drive: https://drive.google.com/drive/folders/1YqglgSbqy-tzIgb09LAQ0RlwOYUm0TtO?usp=sharing
