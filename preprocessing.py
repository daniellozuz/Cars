'''
This script preprocesses all the images and stores them for further use.

Images are loaded, cropped according to ROI, resized to 100x100 and stored into another directory.
'''

import os
import cv2
import scipy.io

mat = scipy.io.loadmat(os.path.join('data', 'cars_annos.mat'))
index = 0
for file_name in os.listdir(os.path.join('data', 'car_ims')):
    image = cv2.imread(os.path.join('data', 'car_ims', file_name), 0)
    print(file_name)
    #cv2.imshow('Car original', image)
    y1 = mat['annotations'][0][index][1][0][0]
    x1 = mat['annotations'][0][index][2][0][0]
    y2 = mat['annotations'][0][index][3][0][0]
    x2 = mat['annotations'][0][index][4][0][0]
    #print(x1, x2, y1, y2)
    #cv2.imshow('Car cropped', image[x1:x2, y1:y2])
    #cv2.imshow('Car resized', cv2.resize(image[x1:x2, y1:y2], (100, 100)))
    cv2.waitKey(100)
    cv2.imwrite(os.path.join('data', 'car_ims_preprocessed', file_name), cv2.resize(image[x1:x2, y1:y2], (100, 100)))
    print(os.path.join('data', 'car_ims_preprocessed', file_name))
    index += 1


print('Annotations:')
print(mat['annotations'])

print('Annotations[0][0]:')
print(mat['annotations'][0][0])

print(mat['annotations'][0][0][0][0])


print('ROI')
print(mat['annotations'][0][0][1][0][0])
print(mat['annotations'][0][0][2][0][0])
print(mat['annotations'][0][0][3][0][0])
print(mat['annotations'][0][0][4][0][0])

print(os.path.join('data', mat['annotations'][0][0][0][0]).replace('/', '\\'))

image = cv2.imread(os.path.join('data', mat['annotations'][0][0][0][0]).replace('/', '\\'), 0)
print(image.shape)
cv2.imshow('A car', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
