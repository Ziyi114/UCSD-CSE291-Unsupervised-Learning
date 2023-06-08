import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tarfile
from path import *

CIFAR_WIDTH = 32
CIFAR_HEIGHT = 32
CIFAR_CHANNEL = 3

# download cifar-10 dataset to dataset_path
def download_cifar10_dataset():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    fileobj = urllib.request.urlopen(url)

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    print('Dataset Downloading ...')
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        tar.extractall(path=dataset_path)

    print('Downloaded CIFAR-10 dataset to ', dataset_path)

# load one batch of training or test data from cifar-10 dataset
def load_one_cifar_batch(
    file_name: str
):
    with open(file_name, 'rb') as f:
        batch_data = pickle.load(
            f, encoding='bytes'
        )
        batch_data[b"data"] = batch_data[b"data"]
    
        return batch_data[b"data"], batch_data[b"labels"]

# load cifar10 dataset and split it into training and test datasets
def load_cifar10_dataset(
    dataset_path: str = cifar10_path,
    subset_train: int = 50000,
    subset_test: int = 10000
):
    # load training set
    x_train = []
    y_train = []
    for i in range(1, 6):
        x_batch, y_batch = load_one_cifar_batch(
            os.path.join(dataset_path, "data_batch_{}".format(i))
        )
        x_train.append(x_batch)
        y_train.append(y_batch)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    # load test set
    x_test, y_test = load_one_cifar_batch(
        os.path.join(dataset_path, "test_batch")
    )
    y_test = np.array(y_test)

    # return the first n images of the training set and test set
    dataset = {
        "x_train": x_train[:subset_train],  # [50000, 3072]
        "y_train": y_train[:subset_train],  # [50000, ]
        "x_test": x_test[:subset_test],     # [10000, 3072]
        "y_test": y_test[:subset_test]      # [10000, ]
    }

    # resize image data to have shape [n, channel, width, height]
    dataset["x_train"] = dataset["x_train"].reshape(
        (-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT)
    )
    dataset["x_test"] = dataset["x_test"].reshape(
        (-1, CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT)
    )

    return dataset

# compute the mean and standard deviation image for cifar-10 dataset
def get_cifar10_mu_std_img():
    # These are pre-computed channel wise mean and the standard deviation for the CIFAR-10 dataset.
    mu = np.array([0.5071, 0.4867, 0.4408])
    std = np.array([0.2675, 0.2565, 0.2761])

    # Creating mean and standard deviation images
    mu_img = np.zeros((CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT), dtype=np.float32)
    std_img = np.zeros((CIFAR_CHANNEL, CIFAR_WIDTH, CIFAR_HEIGHT), dtype=np.float32)
    
    for i in range(mu.shape[0]):
        mu_img[i, ...] = mu[i]
        std_img[i, ...] = std[i]
    
    return mu_img, std_img

# load training and test images and labels
def load_dataset_splits():
    # Load the CIFAR-10 dataset as a dictionary
    dataset = load_cifar10_dataset()
    print("======> CIFAR-10 dataset loaded")

    # Check the shape of the data
    # Train data should have 50,000 samples
    # Test data should have 10,000 samples
    print("Training set data shape: ", dataset['x_train'].shape)
    print("Training set label shape: ", dataset['y_train'].shape)
    print("Test set data shape: ", dataset['x_test'].shape)
    print("Test set label shape: ", dataset['y_test'].shape)

    # Split the data into train and test sets.
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_test = dataset['x_test']
    y_test = dataset['y_test']

    return x_train, y_train, x_test, y_test
    
# samples_per_class: how many images to show per class
def visualize_cifar_data(images, labels, samples_per_class=6):
    # The 10 CIFAR-10 classes
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(cifar_classes)

    # set default size of plots
    plt.rcParams['figure.figsize'] = (20.0, 16.0)  
    
    for cls_index, cls_name in enumerate(cifar_classes):
        idxs = np.flatnonzero(labels == cls_index)
        selected_idxs = np.random.choice(idxs, samples_per_class, replace=False)

        # iterate over selected images in the current class
        for i, idx in enumerate(selected_idxs):
            plt_idx = i * num_classes + cls_index + 1 # each column corresponds to one class
            plt.subplot(samples_per_class, num_classes, plt_idx)
            # to call imshow, image data should have shape: [n, width, height, channel]
            plt.imshow(images[idx] / 255.0)
            plt.axis('off')
            if i == 0:
                plt.title(cls_name)
    
    plt.show()

def normalize(X, mu = None, std = None):
    # Divide by 255 to make the values of the image array in [0, 1]
    X /= 255.0
    
    if std is None:
        std = np.std(X, axis =0)
    if mu is None:
        mu = np.mean(X, axis =0)
    
    # Normalize the images
    return (X - mu) / std