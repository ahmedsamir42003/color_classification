import random
import warnings
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import precision_score,recall_score
from colorthief import ColorThief as ct
warnings.simplefilter('ignore')

# dataset link: https://www.kaggle.com/datasets/adikurniawan/color-dataset-for-color-recognition?resource=download

def unsplitData():
    global dataDir
    for x in os.listdir(dataDir):
        # directory of each color
        dir = dataDir + '\\' + x
        # directory of each color but in the test directory
        test_file_dir = dir + "\\test_images\\"
        # all the directories of the old paths
        imagesOld = []
        # all the directories of the new paths
        imagesNew = []
        # counts the number of files in test_images folder
        counter = 0
        # iterates through the test_images directory
        for y in os.listdir(test_file_dir):
            # appends the new path of the image to a list
            imagesOld.append(dir + "\\" + y)
            # appends the current path of the image to a list
            imagesNew.append(test_file_dir + y)
            # increments the number of images by one
            counter += 1
        # moves the images back from the test_images directory
        for z in range(counter):
            os.rename((imagesNew[z]), (imagesOld[z]))
        # deletes the test_images directory from each color directory
        for iterator in os.listdir(dir):
            try:
                os.removedirs(test_file_dir)
            except FileNotFoundError as e:
                continue


# the data frame that will store the train data
data_train = pd.DataFrame({'r':[],'g':[],'b':[],'color':[]})
data_train = data_train.astype({'r':int,'g':int,'b':int})

# the data frame that will store the test data
data_test = pd.DataFrame({'r':[],'g':[],'b':[],'color':[]})
data_test = data_test.astype({'r':int,'g':int,'b':int})

dataDir = r"C:\Users\DELL\Desktop\dm\final(I hope)\training_dataset"
iterator = 0

if 'test_images' in os.listdir(dataDir + '\\' + 'black'):
    unsplitData()

# to split data into training and testing data (17 for training while 8 for testing)
# iterates through the dataset directory
for x in os.listdir(dataDir):
    # directory of each color
    dir = dataDir + '\\' + x
    # directory of each color but in the test directory
    test_file_dir = dir + "\\test_images\\"
    # all the directories of the old paths
    imagesOld = []
    # all the directories of the new paths
    imagesNew = []
    # iterates through the directories of the colors
    for y in os.listdir(dir):
        # makes directory of the test images
        os.makedirs(test_file_dir, exist_ok=True)
        # appends the current path of the image to a list
        imagesOld.append(dir + "\\" + y)
        # appends the new path of the image to a list
        imagesNew.append(test_file_dir + y)
    # generate list of 8 unique random integers
    added = range(0, 8)
    # moves 8 images to the new test directory 
    for z in added:
        os.rename((imagesOld[z]), (imagesNew[z]))


# the outer loop that iterates through all the folders in the main directory
for x in os.listdir(dataDir):
    dir = dataDir + '\\' + x
    number_of_colors = len(os.listdir(dataDir))
    print(iterator,'out of',number_of_colors,'finished')
    iterator += 1
    
    # the inner loop that iterates through all the pictures in a single directory in the main directory and take its color, store it in the data frame
    for y in os.listdir(dir):
        if y == "test_images":
            dir2 = dir + "\\" + y
            # loop to iterate through test images folder for data tester
            for z in os.listdir(dir2):
                color_test = ct(dir2 + '\\' + z).get_color()
                color_test = pd.DataFrame({'r':[color_test[0]],'g':[color_test[1]],'b':[color_test[2]],'color':x})
                data_test = data_test.append(color_test,ignore_index=True)
            continue

        color_train = ct(dir + '\\' + y).get_color()
        color_train = pd.DataFrame({'r':[color_train[0]],'g':[color_train[1]],'b':[color_train[2]],'color':x})
        data_train = data_train.append(color_train,ignore_index=True)

print(iterator,'out of',number_of_colors,'finished')

# print(data.to_string())

# plots the data frame as lines stacked on top of each others to see if there are outliers
def plotData(data,title):
    if title == 'train':
        fig, ax = plt.subplots(figsize=(5,50))
    else:
        fig, ax = plt.subplots(figsize=(5,20))
    for x in range(len(data.r)):
        hexx = '#{:02x}{:02x}{:02x}'.format(data.r[x], data.g[x], data.b[x])
        ax.axhline(x,color=hexx,linewidth=30)
    ax.set_xlim(0, 1)
    ax.xaxis.set_visible(False)

    plt.title(title)
    plt.show()

plotData(data_train,'train')
plotData(data_test,'test')

# checking nulls and duplicates and removing them
print(data_train.isnull().sum())
print(data_test.isnull().sum())
print(data_train.duplicated().sum())
data_train = data_train.drop_duplicates()

data_all = data_train.append(data_test)
x = data_all[['r','g','b']]
y = data_all[['color']]

X_train = data_train[['r','g','b']]
y_train = data_train[['color']]

X_test = data_test[['r', 'g', 'b']]
y_test = data_test[['color']]

# Preprocess the data
y_encoded = pd.get_dummies(y)
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

unsplitData()

print('first model','-'*50)


# Create a new instance of a Sequential model
model = Sequential()

# Add a fully connected layer to the model with 64 neurons and a ReLU activation function
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Add another fully connected layer to the model with a number of neurons equal to the number of unique classes in y_encoded, and a softmax activation function
model.add(Dense(y_encoded.shape[1], activation='softmax'))

# Compile the model using categorical cross-entropy loss, Adam optimizer, and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data for 50 epochs with a batch size of 32, and use validation data to monitor performance
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data to estimate its performance on unseen data
loss, accuracy = model.evaluate(X_test, y_test)

# Print the test accuracy of the model as a percentage
print('Test accuracy: %.2f' % (accuracy*100))

# Calculate precision and recall
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test.values, axis=1)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print('Precision: %.2f' % precision)
print('Recall: %.2f' % recall)

# Generate classification report
target_names = y_test.columns
print('Confusion matrix:\n', tf.math.confusion_matrix(y_true,y_pred) )
print('Report: \n',classification_report(y_true, y_pred, target_names=target_names))


print('second model','-'*50)
y_train = y_train.idxmax(axis=1)
y_test = y_test.idxmax(axis=1)

# making the model and training it
Logisticmodel = LogisticRegression(solver = 'liblinear',random_state=0)
Logisticmodel.fit(X_train,y_train)

# predicting the data and calculating the metrics of the model
y_pred = Logisticmodel.predict(X_test)
score = Logisticmodel.score(X_test,y_test)
conf_m = confusion_matrix(y_test,y_pred)
report = classification_report(y_test,y_pred)
print('Accuracy: ', score*100)
print('confusion matrix:\n', conf_m)
print('report: \n', report)

# uncomment the next three lines and enter your image path where it says to test the model on an image of your choice.
testImg = ct(r"C:\Users\DELL\Desktop\Youtube_logo.png").get_color()
print("logistic output:",Logisticmodel.predict([[testImg[0],testImg[1],testImg[2]]])[0])
print("ANN output:",target_names[np.argmax(model.predict([[testImg[0],testImg[1],testImg[2]]]),axis=1)][0])