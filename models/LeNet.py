# this file contains the functions for building, training and optimizing a LeNet neural network for digit recognition

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from utilities import OptunaReportCallback
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from tqdm.notebook import tqdm

def LeNet_optimize(datasets, search_space, batch_size, epochs, n_trial, study_name, verbose, storage_url):
    """
    Optimize the hyperparameters of a LeNet neural network using Optuna.

    Args:
        datasets (dict): Dictionary containing training and validation datasets.
        search_space (dict): Hyperparameter search space for Optuna.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train during each trial.
        n_trial (int): Number of Optuna trials to run.
        study_name (str): Name for the Optuna study.
        verbose (int): Verbosity mode for training.
        storage_url (str): URL for persistent Optuna storage (e.g., database).

    Returns:
        optuna.study.Study: The completed Optuna study object with optimization results.
    """
    # definition of the response function for the optimization of the hyperparameters
    def LeNet_objective(trial):
        """
    Objective function for optimizing LeNet hyperparameters with Optuna.

    This function defines the hyperparameter search space, builds the model
    with trial-specific parameters, trains it on the provided dataset,
    and returns the validation accuracy for Optuna to optimize.

    Args:
        trial (optuna.trial.Trial): A single Optuna trial object used to sample hyperparameters.

    Returns:
        float: The maximum validation accuracy achieved during training.
    """
        # creating the dictionary for the hyperparameters of the LeNet
        LeNet_params = {}
        
        # setting the parameters space of the convoluational layers
        LeNet_params["kernel size 1"] = trial.suggest_categorical("kernel size 1", search_space["kernel size 1"])
        LeNet_params["kernel size 2"] = trial.suggest_categorical("kernel size 2", search_space["kernel size 2"])
        LeNet_params["n filter 1"] = trial.suggest_categorical("n filter 1", search_space["n filter 1"])
        LeNet_params["n filter 2"] = trial.suggest_categorical("n filter 2", search_space["n filter 2"])
        LeNet_params["l2 regularizer conv"] = trial.suggest_float("l2 regularizer conv", *search_space["l2 regularizer conv"])
        
        # setting the parameters of the pooling layers
        LeNet_params["pool size 1"] = trial.suggest_categorical("pool size 1", search_space["pool size 1"])
        LeNet_params["pool size 2"] = trial.suggest_categorical("pool size 2", search_space["pool size 2"])
        
        # setting the parameters of the dense layers
        LeNet_params["dense size 1"] = trial.suggest_categorical("dense size 1", search_space["dense size 1"])
        LeNet_params["dense size 2"] = trial.suggest_categorical("dense size 2", search_space["dense size 2"])
        LeNet_params["l2 regularizer dense"] = trial.suggest_float("l2 regularizer dense", *search_space["l2 regularizer dense"])
        
        # setting the activation function of the convolutional layers
        LeNet_params["activation function conv"] = activation_function(trial.suggest_categorical("activation function conv", search_space["activation function conv"]))
        
        # setting the activation function of the dense layers
        LeNet_params["activation function dense"] = activation_function(trial.suggest_categorical("activation function dense", search_space["activation function dense"]))
        
        # creating and training of the LeNet neural network
        model = LeNet(LeNet_params)
        
        # setting the parameters of the early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")
        
        # setting the parameters of the intermediate values callback
        intermediate_values = OptunaReportCallback(trial)
        
        # training of the LeNet neural network
        history = model.fit(x=datasets["train"][0], y=datasets["train"][1], batch_size=batch_size, epochs=epochs, validation_data=(datasets["val"][0], datasets["val"][1]), callbacks=[early_stop, intermediate_values], verbose=verbose)
        
        # computing the accuracy on the training set
        train_accuracy = history.history["accuracy"][np.argmax(history.history["val_accuracy"])]

        # computing the accuracy on the validation set
        val_accuracy = max(history.history["val_accuracy"])
        
        return val_accuracy
        
    # setting the url for saving the study
    storage_url = storage_url
        
    # creating the Optuna study
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_url, sampler=optuna.samplers.TPESampler())
    
    # optimizing the hyperparameters of the LeNet neural network
    study.optimize(LeNet_objective, n_trials=n_trial, show_progress_bar=True)
    
    return study



def LeNet_predict(datasets, model, log_dir, batch_size, epochs, verbose):
    """
    Train the LeNet model and return predictions on the test set.

    Args:
        datasets (dict): Dictionary containing training, validation, and test datasets.
        model (tf.keras.Model): A compiled LeNet model.
        log_dir (str): Directory path for storing TensorBoard logs.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        verbose (int): Verbosity mode for training.

    Returns:
        tuple: (trained model, training accuracy, validation accuracy, training history, predictions on test set)
    """
    # setting the parameters of the early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")
        
    # setting the parameters of the tensorboard callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # training of the LeNet neural network
    history = model.fit(x=datasets["train"][0], y=datasets["train"][1], batch_size=batch_size, epochs=epochs, validation_data=(datasets["val"][0], datasets["val"][1]), callbacks=[early_stop, tensorboard], verbose=verbose)

    # computing the accuracy on the training set
    train_accuracy = history.history["accuracy"][np.argmax(history.history["val_accuracy"])]

    # computing the accuracy on the validation set
    val_accuracy = max(history.history["val_accuracy"])
    
    # using the model for predictions
    y_pred = model.predict(x=datasets["test"])

    return model, train_accuracy, val_accuracy, history.history, y_pred



def LeNet(LeNet_params):
    """
    Build and compile a LeNet-style convolutional neural network using the provided parameters.

    Args:
        LeNet_params (dict): Dictionary containing hyperparameters for model architecture and regularization.

    Returns:
        tf.keras.Model: A compiled LeNet model ready for training.
    """
    # creation of the LeNet neural network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(kernel_size=(LeNet_params["kernel size 1"],LeNet_params["kernel size 1"]), filters=LeNet_params["n filter 1"], padding="same", strides=(1,1), activation=LeNet_params["activation function conv"], kernel_regularizer=tf.keras.regularizers.l2(l2=LeNet_params["l2 regularizer conv"])),
        tf.keras.layers.AveragePooling2D(pool_size=LeNet_params["pool size 1"], padding="valid", strides=(1,1)),
        tf.keras.layers.Conv2D(kernel_size=(LeNet_params["kernel size 2"],LeNet_params["kernel size 2"]), filters=LeNet_params["n filter 2"], padding="same", strides=(1,1), activation=LeNet_params["activation function conv"], kernel_regularizer=tf.keras.regularizers.l2(l2=LeNet_params["l2 regularizer conv"])),
        tf.keras.layers.AveragePooling2D(pool_size=LeNet_params["pool size 2"], padding="valid", strides=(1,1)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=LeNet_params["dense size 1"], activation=LeNet_params["activation function dense"], kernel_regularizer=tf.keras.regularizers.l2(l2=LeNet_params["l2 regularizer dense"])),
        tf.keras.layers.Dense(units=LeNet_params["dense size 2"], activation=LeNet_params["activation function dense"], kernel_regularizer=tf.keras.regularizers.l2(l2=LeNet_params["l2 regularizer dense"])),
        tf.keras.layers.Dense(units=10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(l2=LeNet_params["l2 regularizer dense"]))
    ])
    
    # compilation of the LeNet neural network
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    
    return model



def LeNet_performance(datasets, LeNet_params, batch_size, epochs, n_samples):
    """
    Evaluate LeNet model performance across multiple training runs to assess robustness.

    Args:
        datasets (dict): Dictionary containing training, validation, and test_eval datasets.
        LeNet_params (dict): Dictionary of model hyperparameters.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        n_samples (int): Number of repeated training/evaluation cycles to perform.

    Returns:
        dict: Dictionary containing averaged metrics including accuracy, ROC AUC, and confusion matrices.
    """
    # initialization of the arrays for the temporary results (accuracy)
    train_acc = np.zeros(shape=n_samples, dtype=np.float64)
    val_acc = np.zeros(shape=n_samples, dtype=np.float64)
    test_acc = np.zeros(shape=n_samples, dtype=np.float64)
    
    # initialization of the arrays for the temporary results (roc auc scores)
    roc_auc_scores = np.zeros(shape=(n_samples, 10), dtype=np.float64)
    
    # initialization of the arrays for the temporary results (confusion matrix)
    conf_matrices = np.zeros(shape=( n_samples, 10, 10))
    
    # setting the parameters of the early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")
    
    # wrapping the code for the progress bar
    with tqdm(total=n_samples) as pbar:
        # for running on the samples
        for n in range(n_samples):
            # updating the description of the progress bar
            task_name = f"sample {n+1} of {n_samples}: training..."
            pbar.set_description(task_name)
            
            # creating the LeNet neural network
            model = LeNet(LeNet_params)
            
            # training of the LeNet neural network
            history = model.fit(x=datasets["train"][0], y=datasets["train"][1], batch_size=batch_size, epochs=epochs, validation_data=(datasets["val"][0], datasets["val"][1]), callbacks=[early_stop], verbose=0)
            
            # updating the description of the progress bar
            task_name = f"sample {n+1} of {n_samples}: evaluation..."
            pbar.set_description(task_name)

            # computing the accuracy on the training set
            train_acc[n] = history.history["accuracy"][np.argmax(history.history["val_accuracy"])]

            # computing the accuracy on the validation set
            val_acc[n] = max(history.history["val_accuracy"])
            
            # using the model for predictions
            y_pred = model.predict(x=datasets["test_eval"][0], verbose=0)
            
            # computing the accuracy on the test set
            test_acc[n] = accuracy_score(y_true=datasets["test_eval"][1], y_pred=np.argmax(y_pred, axis=1))
            
            # computing the roc auc scores
            roc_auc_scores[n,:] = roc_auc_score(y_true=datasets["test_eval"][1], y_score=y_pred, multi_class="ovr", labels=np.arange(10), average=None)
            
            # computing the confusion matrix
            conf_matrices[n,:,:] = confusion_matrix(y_true=datasets["test_eval"][1], y_pred=np.argmax(y_pred, axis=1), labels=np.arange(10), normalize="true")
            
            # updating the preogress bar
            pbar.update(1)
            
    # creating the dictionary for the results
    results = {}
    
    # computing the average value and the standard deviation of the accuracy
    results["accuracy"] = {"train": [np.mean(train_acc), np.std(train_acc)],
                           "val": [np.mean(val_acc), np.std(val_acc)],
                           "test": [np.mean(test_acc), np.std(test_acc)]}
    
    # computing the average value and the standard deviation of the roc auc scores
    results["roc auc"] = {"mean": np.mean(roc_auc_scores, axis=0), "std": np.std(roc_auc_scores, axis=0)}
    
    # computing the average value and the standard deviation of the normalized confusion matrix
    results["confusion matrix"] = {"mean": np.mean(conf_matrices, axis=0), "std": np.std(conf_matrices, axis=0)}
    
    return results



def activation_function(function_code):
    """
    Retrieve a Keras activation function based on a given identifier string.

    Args:
        function_code (str): Name of the activation function ('sigmoid', 'tanh', 'relu', 'elu', 'gelu').

    Returns:
        function: Corresponding TensorFlow/Keras activation function.

    Raises:
        ValueError: If an unrecognized activation function code is provided.
    """
    # selection of the activation function
    if function_code == "sigmoid":
        return tf.keras.activations.sigmoid
    elif function_code == "tanh":
        return tf.keras.activations.tanh
    elif function_code == "relu":
        return tf.keras.activations.relu
    elif function_code == "elu":
        return tf.keras.activations.elu
    elif function_code == "gelu":
        return tf.keras.activations.gelu
    else:
        raise ValueError("activation function not recognized")
    
    
    
def plot_history(history, plot_size, classifier_name):
    """
    Plot training and validation accuracy over epochs.

    Args:
        history (dict): Dictionary containing 'accuracy' and 'val_accuracy' history.
        plot_size (tuple): Size of the matplotlib plot.
        classifier_name (str): Name of the classifier for plot title.

    Returns:
        None
    """
    # plotting the history of the training procedure
    plt.figure(figsize=plot_size)
    plt.plot(np.arange(len(history["accuracy"])), history["accuracy"], color="red", label="training accuracy")
    plt.plot(np.arange(len(history["val_accuracy"])), history["val_accuracy"], color="blue", label="validation accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(f"training history of the {classifier_name}")
    plt.legend()
    plt.grid()