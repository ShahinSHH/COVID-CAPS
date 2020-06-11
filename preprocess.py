# Saving images and labels into numpy arrays

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set parameters here 
INPUT_SIZE = (224, 224)
mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}

# train/valid/test .txt files
train_filepath = 'train_split_v3.txt'
valid_filepath = 'valid_split_v3.txt'
test_filepath = 'test_split_v3.txt'

# load in the train and test files
file = open(train_filepath, 'r') 
trainfiles = file.readlines()
file = open(valid_filepath, 'r')
validfiles = file.readlines() 
file = open(test_filepath, 'r')
testfiles = file.readlines()

print('Total samples for train: ', len(trainfiles))
print('Total samples for valid: ', len(validfiles))
print('Total samples for test: ', len(testfiles))

# Total samples for train:  5310
# Total samples for test:  639

# load in images
# resize to input size and normalize to 0 - 1
x_train = []
x_valid = []
x_test = []
y_train = []
y_valid = []
y_test = []


# Create ./data/test - ./data/train - ./data/valid directories yourself
for i in range(len(testfiles)):
    test_i = testfiles[i].split()
    imgpath = test_i[1]
    img = cv2.imread(os.path.join(r'./data/test', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_test.append(img)
    y_test.append(mapping[test_i[2]])

print('Shape of test images: ', x_test[0].shape)

for i in range(len(validfiles)):
    valid_i = validfiles[i].split()
    imgpath = valid_i[1]
    img = cv2.imread(os.path.join(r'./data/valid', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_valid.append(img)
    y_valid.append(mapping[valid_i[2]])

print('Shape of valid images: ', x_valid[0].shape)

for i in range(len(trainfiles)):
    train_i = trainfiles[i].split()
    imgpath = train_i[1]
    img = cv2.imread(os.path.join(r'./data/train', imgpath))
    img = cv2.resize(img, INPUT_SIZE) # resize
    img = img.astype('float32') / 255.0
    x_train.append(img)
    y_train.append(mapping[train_i[2]])

print('Shape of train images: ', x_train[0].shape)

# Shape of test images:  (224, 224, 3)
# Shape of train images:  (224, 224, 3)
# export to npy to load in for training
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_valid.npy', x_valid)
np.save('y_valid.npy', y_valid)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

