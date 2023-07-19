# README File - Color Detection Project

## Table of Contents
* [Introduction](#introduction)
* [Dataset](#dataset)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Usage](#usage)
* [Output](#output)
* [Conclusion](#conclusion)

## Introduction:
This project is about building an application that recognizes the color of an image by evaluating which color is most similar to the selected color value. The project uses Python programming language and a labeled dataset of available colors to train machine learning models that can detect colors easily. The project is useful for people who have difficulty remembering or naming colors, or for those who need to identify colors for various purposes.

## Dataset:
The dataset used in this project is available on Kaggle and consists of images of various colors. The dataset includes 10 color categories with 25 images per category, resulting in a total of 250 images. The dataset link is provided in the code.
[download dataset](https://www.kaggle.com/datasets/adikurniawan/color-dataset-for-color-recognition?resource=download)

## Dependencies:
The project requires the following dependencies to be installed:
- OpenCV (cv2)
- Matplotlib
- Pandas
- Scikit-learn
- NumPy
- Tensorflow
- Keras
- Colorthief

## Installation:
To install the dependencies, you can use the following command:
```python
pip install opencv-python matplotlib pandas scikit-learn numpy tensorflow keras colorthief
```

## Usage:
1. Download the dataset from the provided link and extract it to a directory.
2. Open the Python file and change the value of the 'dataDir' variable to the directory where the dataset is stored.
3. Run the code to train the machine learning models.
4. After training, you can test the models by uncommenting the last three lines of code and entering the path of your image in the 'ct()' function.

## Code:
- First thing we thought about another way of splitting the data instead of the already known function so we made a function that creates a test directory in each color directory and transferes some of the image there to use as test images
- Now we iterate through all images in each color directory and use the colortheif library to extract the dominant color from the picture we then store its RGB values in two dataframes, one for test another for train
- We then check if the data contains nulls or duplicates. Since we extracted colors from images it can't be null but there are duplicates so we remove them
- But how to check if we extracted the correct colors from images? well, we made a plot that plots each color as a horizontal line and stores them in a subplot
- Now it's time for the models
- First model was a neural network and we had to encode the Y to be able to use it in the neural network
- We made two hidden layers
- Now we are good to go, so we fit the model and print the report of it
- For the second model we used logistic regression
- But first we had to return Y to its original state
- We repeat what we did at the first model
- Now we're done. All what's left is to test on images of our own if we want. We left three lines of codes for you to check the model.

## Output:
The project outputs the following:
- A dataframe that stores all the colors extracted from each image.
- The accuracy, confusion matrix, and classification report of two machine learning models.
- The predicted color of the selected image.

## Conclusion:
This project demonstrates how machine learning can be used to detect colors from images. The trained models can be improved further by adding more data or using other techniques. This project can be extended to build a mobile or web application for color recognition.

## Team Members

- [Hazem Mohamed](https://github.com/hazemxiii)
- [Ahmed Samir](https://github.com/ahmedsamir42003)
- [Ahmed Salah](https://github.com/Ahmed-1920)
- [Marwan Ashraf](https://github.com/S7Mario221)
- [Ahmed Nasser](https://github.com/ahmednasser111)
