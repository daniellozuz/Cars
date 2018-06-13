import os
import yaml
import pprint


CONFIGS = [
    # '1.yml',
    # '2.yml',
    # '3.yml',
    '4.yml',
    '5.yml',
    '6.yml',
]


for config in CONFIGS:
    print(f'Simulating {config} with the following parameters:')
    params = yaml.load(open(os.path.join('config', config), 'r'))
    pprint.pprint(params)
    print('Preprocessing...')
    os.system('python3 preprocessing.py ' + config)
    print('Training...')
    os.system('python3 mnist_conv2d_medium_tutorial/train.py ' + config)
    print('Evaluating...')
    os.system('python3 mnist_conv2d_medium_tutorial/evaluate.py ' + config)
