# TC3002B-IA
José Manuel Medina - A01706212

**NOTE:** This is the English version of this file, for the Spanish version see `README-ES.md`

**NOTE:** The dataset can be found in the following Google Drive folder: https://drive.google.com/drive/folders/1YqglgSbqy-tzIgb09LAQ0RlwOYUm0TtO?usp=sharing

**NOTE:** The _notebook_ with all the models can be found in the file `modelo_base_smile_final.ipynb`.

# Smile detection in human faces model
Using A.I. technologies, the goal is to create a model that can recognise if a person is smiling or not.

There are two classes:
- smile
- non_smile

## Dataset description
The smiles dataset was obtained from Kaggle and can be found here: [Smiling or Not | Face Data](https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data) authored by Chazzer.
Inside the dataset we can find the following folders:
- `smile` - 600 64x64 px images
- `non_smile` - 603 64x64 px images
- `test` - ≈12,000 ±100 images, _without classification_, 64x64 px

For practicality we will only take the files in the folders `smile` and `non_smile`.

## Data preprocessing
Once the dataset was selected the data preprocessing took place, separating the training and testing data.
Inside the fodler `model` the new folders `train` and `test` were created with the original dataset 
classes: `smile` and `non-smile`.

The data was split in an 80-20, which means that 80% of the data were destined to the `train` folder and the
remaining 20% to `test`, just as it was suggested in the infographic _"Building the Machine Learning Model"_ by Nantasenamat, C. (2020).

Due to the images being organised alphabetically (the images are named after the person), the first 120 elements of `smile` and `non_smile` were taken to carry out the
original split for `train` and `test`, yielding a total of 240 images, which is roughly ≈20% of all total elements (1,203).

A further split was carried out with the `test` data, half of the images inside this folder were destined to `validation`, leaving us with an 80% `train`, 10% `test`, and 10% `validation` split.

Besides the split, the following scaling techniques were applied: _rescale, rotation_range, width_shift_range, shear_range, horizontal_flip_
this way the model can be provided with more data to train without altering the images performance.

- rescale: Was used to convert the image pixel range from `[0,255] to [0,1]`. This is of great use to normalize the entries and avoid possible biases in each of the image's pixel count.
- rotation_range: As the name says, this is a rotation range in the angles, which means that it rotates the entries, broadening the scenarios which the model could encounter. In this case a `40 degree` range was used, going beyond this range could alter the smiles pattern in the faces and could cause wrong interpretations of the entries.
- width_shift_range: This was used to displace the images on their X axis, left or right depending on the value. In this case we used a `0.1` value due to the nature of dealing with small sized images and because a greater shift would cause the facial expression to be lost from the frame.
- shear_range:This could be said to be a "pulling" of the image as if it was being stretched between two opposite crossed points. In our case we used a value of `0.2` due to the limitations of using small sized images.
- horizontal_flip: Image mirroring, in our case it was set to `True`.

## Initial model
Initially, the model began as a binary sequential model and it consists of seven layers:
- 2 Conv2D layers using `ReLu` as activation algorithm
- 2 MaxPool2D layers between every Conv2D
- 1 Flatten layer
- 1 64 node Dense layer using `ReLu` activation
- 1 Dense layer with 1 node using `sigmoid` activation (due to the binary nature of the model)

It was opted to use an architecture similar to that of a CNN based on a paper written by Guo, X. & Polaina, L. F. (2018), it was also opted to use a Conv2D layer followed by a MaxPool2D with the `ReLu` activation algorithm as suggested in a paper by Malallah, F. et al. (2020). The final Dense layer with the `sigmoid` activation algorithm was used because it returns values between 1 and 0 and thus is of great use in binary classification.

### Metrics and considerations
For the initial model we used the following:
- Binary Cross-Entropy: Used as a _Loss_ function. Measures the lack of simility between actual _labels_ and the predicted positive values (1 pos. 0 neg.) whilst penalizing the predictions that have high certainty but are wrong. This concept can apply to more than two classes (_Categorical_) but in this case it is ideal to use the binary fuction.
- Accuracy: Method to measure the model's performance. Counts predictions where the value equals to the real value.
- Loss: Sum of the errors that were comitted on each of the training set's entries. Accounts for the probabilities or uncertainty of a prediction based on how much it varies from the real value.
- RMSProp: Optimization method. _Root Mean Square Propagation_. This optimization method was used based on the paper by López-Sánchez, M. et al. (2021) which compares the performance of SGD, RMSProp, and Adam. As a sidenote, RMSProp had intermediate results in the paper and Adam had the best performance, that being said, for the first model RMSProp is being used as a benchmark.

### Initial evaluation
The initial model yielded an _Accuracy_ of 92% and an _F1 Score_ of 92%. At first glance and with the reported results in Guo, X. & Polaina, L. F. (2018) paper, our model has an _Accuraacy_ percentage that is comparable to that of a model using _HOG (Labelled + unlabelled)_ as _Features_ and a SVM _Classifier_, as it's ±92%.
Considering that the _validation loss_ (0.2213) is similar to that of _train loss_ (0.2935) and that _train accuracy) (88.99), _validation accuracy_ (90%), and _test accuracy_ (92.25%) are similar, we can assume that the model seem to have been correctly trained. That being said, the _accuracy_ percentage could rise if some of the improvements proposed in the cited papers are implemented, but for a first iteration the model seems to have a good start.

#### Proposed changes for the next model
- Switch RMSProp for Adam

## Second model
The second model is a binary sequential model consisting of seven layers: 
- 2 Conv2D layers using `ReLu` as activation algorithm
- 2 MaxPool2D layers between every Conv2D
- 1 Flatten layer
- 1 64 node Dense layer using `ReLu` activation
- 1 Dense layer with 1 node using `sigmoid` activation (due to the binary nature of the model)

The model structure is the same as the initial model, but for experimentation purposes the _optimizer_ was changed from `RMSprop` to `Adam`. The Conv2D layers also now have 10 filters in the first layer and 15 in the second layer.
The dense layer went from 64 -> 128 perceptrons.

### Initial evaluation
The model yields similar results to those of the first model, with an _Accuracy_ of 90% and and _F1 Score_ of 89%. At first glance there doesn't seem to be any significant improvements with the small changes to the model.
_Validation loss_ is 0.3588 and _train loss_ is 0.2353.
_Train accuracy_ is 90%, _validation accuracy_ is 90%, and _test accuracy_ is 90%.

#### Proposed changes for the next model
- The model in the paper written by Guo, X. & Polaina, L. F. (2018) proposes using a VGG-16 layer, it would be interesting to add it to our model and see how it behaves.

## Third model
The third model is also a binary sequential model and it consists of the following layers:
- 16 VGG16 base layers using `imagenet` as _weights_
- 2 Conv2D layers using `ReLu` as activation algorithm
- 2 MaxPool2D layers between every Conv2D
- 1 Flatten layer
- 1 64 node Dense layer using `ReLu` activation
- 1 Dense layer with 1 node using `sigmoid` activation (due to the binary nature of the model)

As in the second model, the Conv2D layer filters go from 10 - 15.
The _optimizer_ is oncer again `Adam`.

### Initial evaluation
This model had a slight `Accuracy` loss with an 87.5% and an _F1 Score_ of 87%. Despite the worse results, if we analise the _train loss_ (0.2824) versus _validation loss_ (0.4384)
we can infer that the model can be slightly adjusted to obtain better results, as it does not seem to display symptoms of having _overfitting_ or _underfitting_.

It would be interesting to see what would happen with enough computing power to train the VGG16 layers as in our model these were set to not be trainable.

## Final evaluation of the models

| Model                     | Accuracy | Precision | Recall | F1 Score |
|---------------------------|----------|-----------|--------|----------|
| V1(CNN + RMSProp)         | 92%      | 85%       | 100%   | 91%      |
| V2(CNN + Adam)            | 90%      | 85%       | 94%    | 89%      |
| V3(CNN + VGG16 + Adam)    | 87.5%    | 85%       | 89%    | 87%      |

In conclusion, the initial model had the best results out of the three, perhaps because it's the one that used the least amount of parameters (as it had less filters than the others in the Conv2D layers).
The complecity of adding 16 additional layers could have influenced the notable shift in _Accuracy_ seen in the third model.

It must be said that the 100% _Recall_ in the V1 model means that there were no false negatives, and thus, we can say that this model has a high sesibility in detecting positive cases.
Another important aspect to mention is that despite López-Sánchez, M. et al. (2021) proposing that the best optimizer for a CNN is `Adam`, there are certain cases where alternatives such as `RMSProp` can
influence the model in a positive way, as was the case in the V1 model.

## References:
- Nantasenamat, C. (2020). _Building the Machine Learning Model_. Recovered from: https://towardsdatascience.com/how-to-build-a-machine-learning-model-439ab8fb3fb1

- Guo, X. & Polaina, L. F. (2018). _Smile Detection in the Wild Based on Transfer Learning_. http://dx.doi.org/10.1109/FG.2018.00107

- Malallah, F., Al-Jubouri, A., Sabaawi, A., Shareef, B., Saeed, M. & Yasen, K. (2020). _Smiling and Non-smiling Emotion Recognition Based on Lower-half Face using Deep-Learning as Convolutional Neural Network_. http://dx.doi.org/10.4108/eai.28-6-2020.2298175

- López-Sánchez, M., Hernández-Torruco, J., Hernández-Ocaña, B. & Chávez-Bosquez, O. (2021, Febrero 10). Comparative Study of Optimizers in the Training of a Convolutional Neural Network in a Binary Recognition Model. _Research in Computing Science, 150(4)_, 73-82. https://rcs.cic.ipn.mx/2021_150_4/Comparative%20Study%20of%20Optimizers%20in%20the%20Training%20of%20a%20Convolutional%20Neural%20Network.pdf   
