# Part 1: Computer vision for classification task

![example-images-from-dataset](../assets/cats_dogs.png)

## 00 Introduction

The goal for this part is to train a computer vision model for classifying if an image consist of either a dog or a cat.
It is important to note that the task of classification assumes that the image contains a clear subject: either one cat or one dog. The model is designed to predict a single class label (e.g., 'cat' or 'dog') for the entire image. This is also known as binary classification. Other types of classification tasks are multi-class classification and multi-label classification tasks.

## 01 Dataset information

The dataset which has been chosen can be found [here](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset). The dataset consists of 25.000 images of cats and dogs.
Inside the folder `PetImages` there are two folders which is called `Cat` and `Dog`. The dataset is split into two folders: `Cat` and `Dog`. Inside the folder `Cat` there are 12.500 images of cats and inside the folder `Dog` there are 12.500 images of dogs.

In computer vision there is something called augmentations and/or transformations. The purpose with transformations is to augment the dataset in order to increase the generalization of the model. This is done by applying random transformations to the images in the dataset. The transformations can be applied to the training dataset, but not to the validation dataset. The validation dataset should be kept as close to the original dataset as possible.

### Task 1

Go to: `dataset.py` and look at the code which prepares the dataset for training.

Try to look into the [documentation of torchvision transforms](https://pytorch.org/vision/stable/transforms.html) and see if you can find a way to apply the following transformations to the dataset `RandomHorizontalFlip`.

**Extra**:

- Are there any other augmentations which could be applied to the dataset?
- Are there any augmentations which should not be applied to the dataset? If so, why?

## 02 Model set-up

The model which has been chosen for this task is a Convolutional Neural Network (CNN). The CNN is a type of neural network which is designed to recognize patterns in images. The CNN is designed to automatically and adaptively learn spatial hierarchies of features from the data. The CNN is designed to take advantage of the 2D structure of an input image.

The CNN consists of multiple layers which are designed to extract features from the input image. The first layer is designed to extract low-level features such as edges and corners. The following layers are designed to extract higher-level features such as shapes and objects.

The CNN is designed to be trained on a dataset which consists of images and labels. The CNN is designed to learn the relationship between the input image and the label. The CNN is designed to minimize the error between the predicted label and the true label.

### Task 2

Go to: `main.py` and look at the code which trains the model.

!!!!TODO: Add tasks

## 03 Model training

To train the model we need to define a loss function and an optimizer. The loss function is designed to measure the error between the predicted label and the true label. The optimizer is designed to minimize the error between the predicted label and the true label.

The loss function which has been chosen for this task is the Binary Cross Entropy loss function. The Binary Cross Entropy loss function is designed to measure the error between the predicted label and the true label. The Binary Cross Entropy loss function is designed to be used for binary classification tasks.

The optimizer which has been chosen for this task is the Adam optimizer. The Adam optimizer is designed to minimize the error between the predicted label and the true label. The Adam optimizer is designed to be used for training deep learning models.

### Task 3

Run the following command to train the model:

```bash
python 01_classification/src/main.py
```
