# Pokemon Classification project

## Description

This project is a classification of pokemon types using a convolutional neural network. The dataset used is the [Pokemon Images Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) from Kaggle.

The dataset contains around 7000 images of 150 different pokemon types of the first generation.
Using this dataset want to classify the pokemon types.

## Setup
All the project is done using google colab.
The folder distribution to start the project is the following:
We only need the OriginalDataset folder to start the project it can be downloaded from the link above. The other folders will be created automatically.
Models and Data folders are empty at the beginning. The Data folder will be filled with the images of the different types of pokemon. The Models folder will be filled with the models created during the project.

The setup of my google drive is the following: https://drive.google.com/drive/folders/1BKEFRxldnjgc0jrphEyrz6n5zletSqqd?usp=sharing

```bash
.
├── base_folder
│   ├── OriginalDataset
│   │   ├── pokemonName1
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   │   └── imageN.jpg
│   │   ├── pokemonName2
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   │   └── imageN.jpg
│   │   ├── ...
│   │   └── pokemonNameN
│   │       ├── image1.jpg
│   │       ├── image2.jpg
│   │       ├── ...
│   │       └── imageN.jpg
│   ├── Models
│   ├── Data
```

## Data preprocessing
### Data structure
The first step is to preprocess the data. To do this, we have created a python script that takes the images from the OriginalDataset folder and creates a new folder with the images of the different types of pokemon. 
I iterate in the original folder and using a dictionary we create a new folder with the different types of pokemon.
The distribution of the new folder is the following:
Each image is renamed with the name of the pokemon and a number to avoid having images with the same name.

```bash
.
├── base_folder
│   ├── OriginalDataset
│   ├── Models
│   ├── Data
│   │   ├── pokemonType1
│   │   │   ├── pokemonName1_1.jpg
│   │   │   ├── pokemonName1_2.jpg
│   │   │   ├── ...
│   │   │   └── pokemonName1_N.jpg
│   │   │   ├── pokemonName2_1.jpg
│   │   │   ├── pokemonName2_2.jpg
│   │   │   ├── ...
│   │   │   └── pokemonName2_N.jpg
│   │   ├── ...
│   │   └── pokemonTypeN
│   │       ├── pokemonName1_1.jpg
│   │       ├── pokemonName1_2.jpg
│   │       ├── ...
│   │       └── pokemonName1_N.jpg
│   │       ├── pokemonName2_1.jpg
│   │       ├── pokemonName2_2.jpg
│   │       ├── ...
│   │       └── pokemonName2_N.jpg
```
### Data distribution
Evaluation of the dataset:
While we are creating the new structure we are also evaluating the dataset. We are evaluating the number of images of each type.
I realized that the dataset is not balanced. There are types of pokemon that have a lot of images and others that have very few.

In the following image we can see the distribution of the dataset.
![Dataset distribution](https://drive.google.com/file/d/1hy7yUy-h9VDR5FbkMejtfsO2QHSnCkIj/view?usp=sharing)

### Data to tensorflow dataset
Once we have the data in the correct structure we have to create the tensorflow dataset. To do this we have created a python script that takes the images from the Data folder and creates the tensorflow dataset.
I decided the following parameters:
- Image size: 100*100. We have decided to use this size because it is a good compromise between the size of the images and the quality of the images.
- Batch size: 32. I try to train the model with different batch sizes and I have seen that this is the best compromise between the speed of the training and the quality of the model. 
This batch is very important because it is the one that will be used in the training of the model. Tensor flow will infered this batch automatically. I tried to train with batch 1 but the training was very slow.
- Shuffle: True. Our dataset is oredered by type of pokemon. We have to shuffle the dataset to avoid that the model learns the order of the dataset. Is important shuffle the dataset beacuse I'm going to split the dataset in train, validation and test and I want that the three datasets have the same distribution.
- Seed: 42. It doesn't need to be explained.
- label_mode: categorical. We have to use categorical because we have more than two classes, this is a multiclass classification.
- labels: 'inferred'. Tensorflow will infer the labels from the folder structure. The folder name is the label of the images, this is the reason why we have to create the folder structure in the previous step.

### Split the dataset
We have decided to split the dataset in train, validation and test. We have used the following distribution:
- Train: 70%
- Validation: 15%
- Test: 15%

We finally got the following number of batches.
- Complete dataset: 215

- Train: 149
- Validation: 32
- Test: 34


## Models

### Save models callback
We have created a callback to save the models.That we explain above.


### About the models
The models are created using tensorflow and keras. All the models are documentated in the notebook, we only explain here the most important models. The other models have small changes in hyperparameters. 

Choosing the best model is a very important step. We have to choose a model that is enough complex to learn the patterns of the dataset.We are try with neuronal networks.
Starting with a simple model and increasing the complexity. 

### Model 1

The first model is a very simple model. We have used the following layers:

- Input layer: The input layer is the size of the images. 
- Flatten layer: This layer is used to flatten the input. The input of the flatten layer is the output of the input layer. The output of the flatten layer is a vector with the size of the input. In this case the output is a vector of 10000 elements.
- Dense layers: We have used 3 dense layers with 256 neurons and relu activation function. The relu activation function is the most used activation function in the hidden layers. We have used 256 neurons because it is a good compromise between the complexity of the model and the speed of the training.
- Output layer: The output layer has the number of neurons equal to the number of classes. In this case we have 150 neurons because we have 150 classes. The activation function is softmax because we have a multiclass classification problem.

### Results model 1

The results of the model are the following:
train acc: 0.1437
val acc: 0.1279
loss train: 7.7808
loss val: 2.7038

The model is not learning. The accuracy is very low and the loss is very high.
The model is not complex enough to learn the patterns of the dataset, we have to increase the complexity of the model. The problem is that our dataset is not very big and we can't create a very complex model. For this reason we I decided to use a pretrained model. 

### Model 2

The second model is a pretrained model. We have used the VGG16 model. We have used the following layers:

- Input layer: The input layer is the size of the images.
- VGG16: We have used the VGG16 model. We have used the weights of imagenet. We have used the imagenet weights because the model is trained with a very big dataset and the weights are very good. We have used the VGG16 model because it is a good compromise between the complexity of the model and the speed of the training.
- Flatten layer: This layer is used to flatten the input. The input of the flatten layer is the output of the VGG16 model. The output of the flatten layer is a vector with the size of the input. In this case the output is a vector of 25088 elements.
- Dense layers: We have used 1 dense layer with 256 neurons and relu activation function. The relu activation function is the most used activation function in the hidden layers.

Using this pretrained model we are looking to take advantage of the weights of the model. The last layers are able to recognize patterns in the images of the imagenet dataset
that are all types of image that we can find in the real world we are interested in recognize colors and animals, beacuse we are classifying pokemon.
The VGG16 model is not able to recognize pokemon types but it is able to recognize colors and animals, for this reason we add a dense layer to learn the patterns of the pokemon types.

### Results model 2

**Train Accuarcy: 0.9386**
**Test Acuarcy :  0.785**
Batch_size: 32
Epochs: 10-
- loss: 0.1619
- val_loss: 1.0303
- val_accuracy: 0.8135

train_results:
https://drive.google.com/file/d/1huTzOubNrabmEod0Q2iaLKvGh0MpODlV/view?usp=sharing

Exist a big difference between the train accuracy and the test accuracy. This is a sign of overfitting. The model is learning the patterns of the train dataset but it is not able to generalize to the test dataset. But, compared to the first model, the accuracy is much higher and the loss is much lower. A test accuracy of 0.785 is a good result. We have to try to reduce the overfitting.

## Deploy

We saved the second model in a .h5 file. We have created a python script that loads the model and predicts the type of pokemon of an image. The script takes an image as input and returns the type of pokemon of the image. The notebook DeployPokemonClassification.ipynb contains the script. We must add some pokemons to a folder and then read it and predict the type of pokemon.

