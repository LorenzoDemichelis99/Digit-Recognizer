# this file contains the functions for loading and preprocessing the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show_image(dataset_array, plot_code, image_index, plot_size, y_pred=None):
    """
    Displays a single image from a dataset with optional label or prediction information.

    Depending on the selected `plot_code`, this function visualizes an image along with
    its true label, index, or predicted label.

    Args:
        dataset_array (np.ndarray): The dataset array from which to extract the image.
            - For "show": expected shape is (N, 785) where the first column is the label.
            - For "explore" and "predict": expected shape is (N, 784).
        plot_code (str): Display mode. One of:
            - "show": Displays the image with its true label (for labeled data).
            - "explore": Displays the image only (no label), for visual inspection.
            - "predict": Displays the image with its predicted label.
        image_index (int): Index of the image to display.
        plot_size (tuple): Figure size as (width, height).
        y_pred (np.ndarray, optional): Prediction array (used only when `plot_code == "predict"`).

    Raises:
        ValueError: If `plot_code` is not one of the supported values.
    """
    # checking the plot code
    if plot_code not in ["show", "explore", "predict"]:
        raise ValueError("plot code not recognized")
    
    if plot_code == "show":
        # selecting the label and the image
        image_label = dataset_array[image_index,0]
        image_array = np.reshape(dataset_array[image_index,1:], (28,28))
        
        # plotting the image
        plt.figure(figsize=plot_size)
        plt.imshow(image_array, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"image index: {image_index} - label: {image_label}")
    elif plot_code == "explore":
        # selecting the image
        image_array = np.reshape(dataset_array[image_index], (28,28))
        
        # plotting the image
        plt.figure(figsize=plot_size)
        plt.imshow(image_array, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"image index: {image_index}")
    else:
        # selecting the image
        image_label = np.argmax(y_pred[image_index,:])
        image_array = np.reshape(dataset_array[image_index], (28,28)) 
        
        # plotting the image
        plt.figure(figsize=plot_size)
        plt.imshow(image_array, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"image index: {image_index} - predicted label: {image_label}")
    
    

def create_dataset(dataset_array, dataset_code):
    """
    Creates a dataset split (features and labels) from a raw dataset array.

    This function reshapes a flat image array into a 3D format (28x28) and
    extracts labels if available. Useful for preparing data for training or inference.

    Args:
        dataset_array (np.ndarray): Input dataset array.
            - For labeled data: shape should be (N, 785), where column 0 is the label.
            - For test data: shape should be (N, 784), with no label column.
        dataset_code (str): One of:
            - "train" or "val": Indicates labeled data with labels in the first column.
            - "test": Indicates unlabeled test data.

    Returns:
        tuple or np.ndarray:
            - If dataset_code != "test": Returns (X, y)
                - X (np.ndarray): Images reshaped to shape (N, 28, 28)
                - y (np.ndarray): Labels array of shape (N,)
            - If dataset_code == "test": Returns only X

    Raises:
        ValueError: If `dataset_code` is not recognized (currently not explicitly raised).
    """
    # checking the dataset code
    if dataset_code not in ["train", "test"]:
        raise ValueError(f"dataset code not recognized - expecting 'train' or 'test' - got {dataset_code}")
    
    if dataset_code != "test":
        # selecting the data
        X = np.reshape(dataset_array[:,1:], (dataset_array.shape[0], 28, 28))

        # selecting the labels
        y = dataset_array[:,0]
        
        return X, y
    else:
        # selecting the data
        X = np.reshape(dataset_array[:], (dataset_array.shape[0], 28, 28))
        
        return X