import os

os.system('python3 preprocessing.py 1.yml')
os.system('python3 mnist_conv2d_medium_tutorial/train.py 1.yml')
os.system('python3 mnist_conv2d_medium_tutorial/evaluate.py 1.yml')
