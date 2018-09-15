'''
Download MNIST (and Car in the future) data
'''

import subprocess


def download_mnist_data():
    file_urls = ['https://pjreddie.com/media/files/mnist_train.csv',
                 'https://pjreddie.com/media/files/mnist_test.csv']
    wget_parameters = ['wget', '-N', '-P', 'data/']
    subprocess.call(wget_parameters + file_urls)


if __name__ == '__main__':
    download_mnist_data()
