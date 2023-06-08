import numpy as np
import os
import torch
from pathlib import Path
import pickle
from tqdm import tqdm

from dataset import get_cifar10_mu_std_img, normalize

from skimage.feature import hog

from vgg_network import vgg11_bn
from path import *



def compute_raw_pixel_features(x_train, x_test):
    raw_pixel_train_features = np.reshape(x_train, (x_train.shape[0], -1))
    raw_pixel_test_features = np.reshape(x_test, (x_test.shape[0], -1))
    print("======> Done with computation of raw pixel features")

    return raw_pixel_train_features, raw_pixel_test_features

def compute_hog_features(x_train, x_test):
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]

    # compute hog features for training images
    hog_train_features = list()
    for i in tqdm(range(num_train_samples)):
        # x_train[i]: [c, w, h] 
        fd = hog(x_train[i], orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), feature_vector=True, channel_axis=0)
        
        hog_train_features.append(fd)
        
    # compute hog features for test images
    hog_test_features = list()
    for i in tqdm(range(num_test_samples)):
        # x_test[i]: [c, w, h] 
        fd = hog(x_test[i], orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), feature_vector=True, channel_axis=0)
        
        hog_test_features.append(fd)
        
    hog_train_features = np.array(hog_train_features)
    hog_test_features = np.array(hog_test_features)

    print("======> Done with computation of HoG features")

    return hog_train_features, hog_test_features

def compute_pretrained_cnn_features(x_train, x_test, mu_img, std_img, layer):
    # Load the VGG11 model with weights pre-trained on CIFAR-10
    deep_model = vgg11_bn(pretrained=True)
    deep_model.eval() 

    return compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer)

def compute_random_cnn_features(x_train, x_test, mu_img, std_img, layer):
    # Create the VGG11 model with random weights
    deep_model = vgg11_bn(pretrained=False)
    deep_model.eval() 

    return compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer)

def compute_cnn_features(deep_model, x_train, x_test, mu_img, std_img, layer):
    
    # normalize image dataset
    x_train_ = normalize(np.copy(x_train).astype(np.float32), mu_img, std_img)
    x_test_ = normalize(np.copy(x_test).astype(np.float32), mu_img, std_img)

    # Compute features in batches
    batch_size = 100 
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]
    assert num_train_samples % batch_size == 0, "Error: The number of training samples should be divisible by the batch size"
    assert num_test_samples % batch_size == 0, "Error: The number of test samples should be divisible by the batch size"

    # compute train features
    x_deep_features_train = []
    num_train_batches = num_train_samples // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_train_batches)):
            # forward NN
            cur_feature_batch = deep_model.extract_features(
                x = torch.tensor(x_train_[i*batch_size : (i+1)*batch_size]),
                layer = layer).detach().numpy()
            x_deep_features_train.append(cur_feature_batch)

    x_deep_features_train = np.array(x_deep_features_train).reshape(num_train_samples, -1) 

    # compute test features
    x_deep_features_test = []
    num_test_batches = num_test_samples // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_test_batches)):
            # forward NN
            cur_feature_batch = deep_model.extract_features(
                torch.tensor(x_test_[i*batch_size :(i+1)*batch_size]),
                layer = layer).detach().numpy()
            x_deep_features_test.append(cur_feature_batch)

    x_deep_features_test = np.array(x_deep_features_test).reshape(num_test_samples, -1) 

    print("======> Done with computation of CNN features")

    return x_deep_features_train, x_deep_features_test

def load_features(sp_feature_path):
    with open(sp_feature_path, 'rb') as f:
        features = pickle.load(f)
        
    print('======> Loaded train and test features from ', sp_feature_path)

    return features["train"], features["test"]
    

def save_features(train_features, test_features, sp_feature_path):
    features = {"train": train_features, "test": test_features}
    
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    
    # save
    with open(sp_feature_path, 'wb') as f:
        pickle.dump(features, f)
    
    print('======> Saved train and test features to ', sp_feature_path)

# compute or load features
def compute_or_load_features(x_train, x_test, feature_type, layer=None):
    assert feature_type in ['raw_pixel', 'hog', 'pretrained_cnn', 'random_cnn'], "Error: Invalid feature type"
    if feature_type in ['raw_pixel', 'hog']:
        assert layer is None, "Error: Layer can only be set when feature type is pretrained_cnn or random_cnn"
    else:
        assert layer in ['last_conv', 'last_fc'], "Error: Invalid layer type"
    
    if layer is None:
        sp_feature_path = os.path.join(feature_path, feature_type+".pkl")
    else:
        sp_feature_path = os.path.join(feature_path, feature_type+"_"+layer+".pkl")
    
    feature_file = Path(sp_feature_path)
    # load features from existing file
    if feature_file.is_file():
        train_features, test_features = load_features(sp_feature_path)
    # compute features
    else:
        if feature_type == "raw_pixel":
            train_features, test_features = compute_raw_pixel_features(x_train, x_test)
        elif feature_type == "hog":
            train_features, test_features = compute_hog_features(x_train, x_test)
        elif feature_type == "pretrained_cnn":
            mu_img, std_img = get_cifar10_mu_std_img()
            train_features, test_features = compute_pretrained_cnn_features(x_train, x_test, mu_img, std_img, layer)
        elif feature_type == "random_cnn":
            mu_img, std_img = get_cifar10_mu_std_img()
            train_features, test_features = compute_random_cnn_features(x_train, x_test, mu_img, std_img, layer)
        else:
            raise NotImplementedError
        
        save_features(train_features, test_features, sp_feature_path)

    print("Training feature shape: ", train_features.shape)
    print("Test feature shape: ", test_features.shape)

    return train_features, test_features

