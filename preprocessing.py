'''
This script preprocesses the testing and training data to be in csv format.

This is for ease of development (the model and stuff is already prepared).
'''

import os
import cv2
import scipy.io
import csv
import random

mat = scipy.io.loadmat(os.path.join('data', 'cars_annos.mat'))

with open(os.path.join('data', 'cars_all.csv'), 'w') as car_file,\
     open(os.path.join('data', 'cars_train.csv'), 'w') as train_file,\
     open(os.path.join('data', 'cars_test.csv'), 'w') as test_file:
    all_writer = csv.writer(car_file)
    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)
    cars_all_dir = os.path.join('data', 'cars_all')
    for index, file_name in enumerate(sorted(os.listdir(cars_all_dir))):
        image = cv2.imread(os.path.join(cars_all_dir, file_name), 0)
        print(file_name)
        y1 = mat['annotations'][0][index][1][0][0]
        x1 = mat['annotations'][0][index][2][0][0]
        y2 = mat['annotations'][0][index][3][0][0]
        x2 = mat['annotations'][0][index][4][0][0]
        label = mat['annotations'][0][index][5][0][0]
        # cv2.imshow('Car cropped', image[x1:x2, y1:y2])
        resized = cv2.resize(image[x1:x2, y1:y2], (28, 28))
        # cv2.imshow('Car resized', resized)
        # cv2.waitKey(100)
        entry = [label] + [item for row in resized for item in row]
        all_writer.writerow(entry)
        writer = random.choice([train_writer, test_writer])
        writer.writerow(entry)

cv2.destroyAllWindows()
