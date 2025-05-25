# this file contains the functions required to compute and plot the learning curve of the LeNet neural network and of the ensemble of neural networks

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.LeNet import LeNet
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm

def learning_curve(training_set_sizes, datasets, n_samples, LeNet_params, batch_size, epochs):
    """
    Computes the learning curve of a single LeNet model by training on increasing subsets 
    of the dataset and evaluating performance.

    For each training set size, the model is trained `n_samples` times with different 
    subsets, and the mean and standard deviation of both training and test accuracy are computed.

    Args:
        training_set_sizes (np.ndarray): Array of floats representing fractions of the training set size (e.g., [0.1, 0.2, ..., 1.0]).
        datasets (dict): Dictionary containing:
            - "train": (X_train, y_train)
            - "val": (X_val, y_val)
            - "test_eval": (X_test, y_test)
        n_samples (int): Number of times to repeat training per subset size for averaging.
        LeNet_params (dict): Parameters to initialize the LeNet model.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.

    Returns:
        tuple:
            - train_acc_avg (np.ndarray): Average training accuracy for each training set size.
            - test_acc_avg (np.ndarray): Average test accuracy for each training set size.
            - train_acc_std (np.ndarray): Standard deviation of training accuracy.
            - test_acc_std (np.ndarray): Standard deviation of test accuracy.
    """
    # initialization of the arrays for the results
    train_acc_avg = np.zeros(shape=training_set_sizes.shape[0], dtype=np.float64)
    train_acc_std = np.zeros(shape=training_set_sizes.shape[0], dtype=np.float64)
    test_acc_avg = np.zeros(shape=training_set_sizes.shape[0], dtype=np.float64)
    test_acc_std = np.zeros(shape=training_set_sizes.shape[0], dtype=np.float64)
    
    # initialization of the arrays for the temporary results
    train_acc_temp = np.zeros(shape=n_samples, dtype=np.float64)
    test_acc_temp = np.zeros(shape=n_samples, dtype=np.float64)
    
    # wrapping the code for the outer progress bar
    with tqdm(total=training_set_sizes.shape[0]) as pbar_out:
        # for running on the training set sizes
        for i in range(training_set_sizes.shape[0]):
            # updating the description of the outer progress bar
            task_name_out = f"size {i+1} of {training_set_sizes.shape[0]}"
            pbar_out.set_description(task_name_out)
            
            # sampling the indices for the training set
            train_indices = sample_without_replacement(n_population=datasets["train"][0].shape[0], n_samples=int(training_set_sizes[i] * datasets["train"][0].shape[0]))
            
            # wrapping the code for the inner progress bar
            with tqdm(total=n_samples) as pbar_in:
                # for running on the samples
                for n in range(n_samples):
                    # updating the description of the inner progress bar
                    task_name_in = f"sample {n+1} of {n_samples}"
                    pbar_in.set_description(task_name_in)
                    
                    # creating the LeNet neural network
                    model = LeNet(LeNet_params)
                    
                    # setting the parameters of the early stopping callback
                    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max")
                    
                    # training of the LeNet neural network
                    model.fit(x=datasets["train"][0][train_indices], y=datasets["train"][1][train_indices], batch_size=batch_size, epochs=epochs, validation_data=(datasets["val"][0], datasets["val"][1]), callbacks=[early_stop], verbose=0)
                    
                    # computing the accuracy on the training set
                    train_acc_temp[n] = model.evaluate(x=datasets["train"][0][train_indices], y=datasets["train"][1][train_indices], verbose=0)[1]

                    # computing the accuracy on the test set
                    test_acc_temp[n] = model.evaluate(x=datasets["test_eval"][0], y=datasets["test_eval"][1], verbose=0)[1]
                    
                    # updating the inner progress bar
                    pbar_in.update(1)
                
            # computing the average value and the standard deviation of the accuracy on the training set for the current size    
            train_acc_avg[i] = np.mean(train_acc_temp)
            train_acc_std[i] = np.std(train_acc_temp)
            
            # computing the average value and the standard deviation of the accuracy on the test set for the current size    
            test_acc_avg[i] = np.mean(test_acc_temp)
            test_acc_std[i] = np.std(test_acc_temp)
            
            # updating the outer progress bar
            pbar_out.update(1)
    
    return train_acc_avg, test_acc_avg, train_acc_std, test_acc_std
                
                
                
def learning_curve_ensemble(training_set_sizes, datasets, n_samples, LeNet_params, batch_size, epochs, ensemble_size):
    """
    Computes the learning curve for an ensemble of LeNet models by training on 
    increasing subsets of the dataset and averaging ensemble predictions.

    For each training set size, an ensemble of models is trained `n_samples` times, and 
    the ensemble prediction is computed by averaging outputs. Results are averaged 
    to obtain the learning curve statistics.

    Args:
        training_set_sizes (np.ndarray): Array of floats representing fractions of the training set size.
        datasets (dict): Dictionary containing:
            - "train": (X_train, y_train)
            - "val": (X_val, y_val)
            - "test_eval": (X_test, y_test)
        n_samples (int): Number of repetitions for each subset size.
        LeNet_params (dict): Parameters used to initialize each LeNet model.
        batch_size (int): Batch size for model training.
        epochs (int): Number of epochs for training.
        ensemble_size (int): Number of models in the ensemble.

    Returns:
        tuple:
            - train_acc_avg (np.ndarray): Average training accuracy for each training set size.
            - test_acc_avg (np.ndarray): Average test accuracy for each training set size.
            - train_acc_std (np.ndarray): Standard deviation of training accuracy.
            - test_acc_std (np.ndarray): Standard deviation of test accuracy.
    """
    # initialization of the arrays for the results
    train_acc_avg = np.zeros(shape=training_set_sizes.shape[0], dtype=np.float64)
    train_acc_std = np.zeros(shape=training_set_sizes.shape[0], dtype=np.float64)
    test_acc_avg = np.zeros(shape=training_set_sizes.shape[0], dtype=np.float64)
    test_acc_std = np.zeros(shape=training_set_sizes.shape[0], dtype=np.float64)
    
    # initialization of the arrays for the temporary results
    train_acc_temp = np.zeros(shape=n_samples, dtype=np.float64)
    test_acc_temp = np.zeros(shape=n_samples, dtype=np.float64)
    
    # wrapping the code for the outer progress bar
    with tqdm(total=training_set_sizes.shape[0]) as pbar_out:
        # for running on the training set sizes
        for i in range(training_set_sizes.shape[0]):
            # updating the description of the outer progress bar
            task_name_out = f"size {i+1} of {training_set_sizes.shape[0]}"
            pbar_out.set_description(task_name_out)
            
            # sampling the indices for the training set
            train_indices = sample_without_replacement(n_population=datasets["train"][0].shape[0], n_samples=int(training_set_sizes[i] * datasets["train"][0].shape[0]))
            
            # wrapping the code for the inner progress bar
            with tqdm(total=n_samples) as pbar_in:
                # for running on the samples
                for n in range(n_samples):
                    # updating the description of the inner progress bar
                    task_name_in = f"sample {n+1} of {n_samples}: training..."
                    pbar_in.set_description(task_name_in)
                    
                    # creating the LeNet neural networks of the ensemble
                    models = [LeNet(LeNet_params) for _ in range(ensemble_size)]
                    
                    # setting the parameters of the early stopping callback
                    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max")
                    
                    # training the networks of the ensemble
                    histories = [models[i].fit(x=datasets["train"][0][train_indices], y=datasets["train"][1][train_indices], batch_size=batch_size, epochs=epochs, validation_data=(datasets["val"][0], datasets["val"][1]), callbacks=[early_stop], verbose=0) for i in range(ensemble_size)]
                    
                    # updating the description of the inner progress bar
                    task_name = f"sample {n+1} of {n_samples}: evaluation..."
                    pbar_in.set_description(task_name)
                    
                    # computing the accuracy on the training set
                    train_acc_temp[n] = np.mean(np.array([histories[i].history["accuracy"][np.argmax(histories[i].history["val_accuracy"])] for i in range(ensemble_size)]))

                    # using the model for predictions
                    y_pred = np.sum(np.array([models[i].predict(datasets["test_eval"][0], verbose=0) for i in range(ensemble_size)]), axis=0)
                    
                    # computing the accuracy on the test set
                    test_acc_temp[n] = accuracy_score(y_true=datasets["test_eval"][1], y_pred=np.argmax(y_pred, axis=1))
                    
                    # updating the inner progress bar
                    pbar_in.update(1)
                
            # computing the average value and the standard deviation of the accuracy on the training set for the current size    
            train_acc_avg[i] = np.mean(train_acc_temp)
            train_acc_std[i] = np.std(train_acc_temp)
            
            # computing the average value and the standard deviation of the accuracy on the test set for the current size    
            test_acc_avg[i] = np.mean(test_acc_temp)
            test_acc_std[i] = np.std(test_acc_temp)
            
            # updating the outer progress bar
            pbar_out.update(1)
    
    return train_acc_avg, test_acc_avg, train_acc_std, test_acc_std



def plot_learning_curve(training_set_sizes, lc_avg_train_acc, lc_avg_test_acc, lc_std_train_acc, lc_std_test_acc, plot_size, classifier_name):
    """
    Plots the learning curve for a classifier, showing training and test accuracy as 
    a function of training set size, along with shaded regions for standard deviation.

    Args:
        training_set_sizes (np.ndarray): Array of training set sizes used in the experiment.
        lc_avg_train_acc (np.ndarray): Average training accuracy per set size.
        lc_avg_test_acc (np.ndarray): Average test accuracy per set size.
        lc_std_train_acc (np.ndarray): Standard deviation of training accuracy.
        lc_std_test_acc (np.ndarray): Standard deviation of test accuracy.
        plot_size (tuple): Size of the plot (width, height).
        classifier_name (str): Name of the classifier (used in plot title).

    Returns:
        None
    """
    # plotting the learning curve
    plt.gcf()
    plt.figure(figsize=plot_size)
    plt.plot(training_set_sizes, lc_avg_train_acc, color="blue", label="training accuracy")
    plt.fill_between(x=training_set_sizes, y1=lc_avg_train_acc - lc_std_train_acc, y2=np.array([min(1.0, lc_avg_train_acc[i] + lc_std_train_acc[i]) for i in range(lc_avg_train_acc.shape[0])]), color="blue", alpha=0.2)
    plt.plot(training_set_sizes, lc_avg_test_acc, color="red", label="test accuracy")
    plt.fill_between(x=training_set_sizes, y1=lc_avg_test_acc - lc_std_test_acc, y2=np.array([min(1.0, lc_avg_test_acc[i] + lc_std_test_acc[i]) for i in range(lc_avg_test_acc.shape[0])]), color="red", alpha=0.2)
    plt.xlabel("size of the training set")
    plt.ylabel("accuracy")
    plt.title(f"Learning Curve of the {classifier_name}")
    plt.legend()
    plt.grid()