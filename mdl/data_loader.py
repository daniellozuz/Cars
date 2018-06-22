import numpy as np


def load_train_data(data_path, num_labels, image_size, validation_size=500, **kwargs):
    """
    Load data. Each row in csv is formatted (label, input)
    :return: 3D Tensor input of train and validation set with 2D Tensor of one hot encoded image labels
    """
    # Data format: 1 byte label, n * n input
    train_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    np.random.shuffle(train_data)
    x_train = train_data[:, 1:]

    # Get label and one-hot encode
    y_train = train_data[:, 0]
    y_train = (np.arange(num_labels) == y_train[:, None]).astype(np.float32)

    # get a validation set and remove it from the train set
    x_train, x_val, y_train, y_val = x_train[0:(len(x_train) - validation_size), :], x_train[(
        len(x_train) - validation_size):len(x_train), :], \
                                     y_train[0:(len(y_train) - validation_size), :], y_train[(
        len(y_train) - validation_size):len(y_train), :]

    # reformat the data so it's not flat
    x_train = x_train.reshape(len(x_train), image_size, image_size, 1)
    x_val = x_val.reshape(len(x_val), image_size, image_size, 1)

    return x_train, x_val, y_train, y_val


def load_test_data(data_path, num_labels, image_size, **kwargs):
    """
    Load test data
    :return: 3D Tensor input of train and validation set with 2D Tensor of one hot encoded image labels
    """
    test_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    np.random.shuffle(test_data)    
    x_test = test_data[:, 1:]

    y_test = np.array(test_data[:, 0])
    y_test = (np.arange(num_labels) == y_test[:, None]).astype(np.float32)

    x_test = x_test.reshape(len(x_test), image_size, image_size, 1)

    return x_test, y_test
