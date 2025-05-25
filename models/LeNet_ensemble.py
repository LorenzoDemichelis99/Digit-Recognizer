# this file contains the functions required to implement and evaluate a deep ensemble built using LeNet neural networks

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.LeNet import LeNet
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from tqdm.notebook import tqdm

def LeNet_ensemble_performance(datasets, LeNet_params, batch_size, epochs, n_samples, ensemble_size):
    """
    Computes the performance metrics of a deep ensemble built using LeNet neural networks.

    The ensemble is trained and evaluated multiple times to compute mean and standard deviation
    of classification accuracy, ROC AUC scores, and the normalized confusion matrix.

    Args:
        datasets (dict): Dictionary containing the training, validation, and test datasets.
        LeNet_params (dict): Dictionary of parameters used to initialize the LeNet models.
        batch_size (int) Batch size used for training.
        epochs (int): Maximum number of training epochs.
        n_samples (int): Number of repetitions for ensemble training to compute statistics.
        ensemble_size (int): Number of LeNet models in each ensemble.

    Returns:
        results (dict): Dictionary containing average and standard deviation of:
            - accuracy on train, validation, and test sets,
            - ROC AUC scores for each class,
            - normalized confusion matrices.
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
    
    # wrapping the code for the outer progress bar
    with tqdm(total=n_samples) as pbar:
        # for running on the samples
        for n in range(n_samples):
            # updating the description of the progress bar
            task_name = f"sample {n+1} of {n_samples}: training..."
            pbar.set_description(task_name)
            
            # creating the LeNet neural networks of the ensemble
            models = [LeNet(LeNet_params) for _ in range(ensemble_size)]
            
            # training the networks of the ensemble
            histories = [models[i].fit(x=datasets["train"][0], y=datasets["train"][1], batch_size=batch_size, epochs=epochs, validation_data=(datasets["val"][0], datasets["val"][1]), callbacks=[early_stop], verbose=0) for i in range(ensemble_size)]
            
            # updating the description of the progress bar
            task_name = f"sample {n+1} of {n_samples}: evaluation..."
            pbar.set_description(task_name)
            
            # computing the accuracy on the training set
            train_acc[n] = np.mean(np.array([histories[i].history["accuracy"][np.argmax(histories[i].history["val_accuracy"])] for i in range(ensemble_size)]))
            
            # computing the accuracy on the validation set
            val_acc[n] = np.mean(np.array([max(histories[i].history["val_accuracy"]) for i in range(ensemble_size)]))
            
            # using the model for predictions
            y_pred = np.sum(np.array([models[i].predict(datasets["test_eval"][0], verbose=0) for i in range(ensemble_size)]), axis=0)

            # normalization of the rows of y_pred
            y_pred = np.apply_along_axis(func1d=lambda p: p / np.sum(p), axis=1, arr=y_pred)
            
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



def LeNet_ensemble_predict(datasets, models, batch_size, epochs):
    """
    Trains the given ensemble of LeNet models and performs predictions on the test set.

    Each model in the ensemble is trained with early stopping, and the final predictions
    are obtained by averaging the softmax outputs of all ensemble members.

    Args:
        datasets (dict): Dictionary containing the training, validation, and test datasets.
        models (list): List of untrained LeNet model instances to be used in the ensemble.
        batch_size (int): Batch size used for training.
        epochs (int): Maximum number of training epochs.

    Returns:
        models (list): The trained ensemble of LeNet models.
        train_accuracy (float): Average training accuracy of the ensemble.
        val_accuracy (float): Average validation accuracy of the ensemble.
        histories (list):  List of training histories for each model in the ensemble.
        y_pred (np.ndarray): Averaged predictions of the ensemble on the test set.
    """
    # initialization of the the accuracies
    train_accuracy = 0.0
    val_accuracy = 0.0
    
    # initialization of the list of the histories
    histories = []
    
    # initialization of the array of the predictions
    y_pred = np.zeros(shape=(datasets["test"].shape[0],10), dtype=np.float64)
    
    # setting the parameters of the early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")
    
    # wrapping the code for the progress bar
    with tqdm(total=len(models)) as pbar:
        # for running on the ensemble
        for n in range(len(models)):
            # updating the description of the progress bar
            task_name = f"training of model {n+1} of {len(models)}"
            pbar.set_description(task_name)
            
            # training of the LeNet neural network
            history = models[n].fit(x=datasets["train"][0], y=datasets["train"][1], batch_size=batch_size, epochs=epochs, validation_data=(datasets["val"][0], datasets["val"][1]), callbacks=[early_stop], verbose=0)
            
            # saving the training history
            histories.append(history.history)

            # updating the accuracy on the training set
            train_accuracy += ((1/len(models)) * history.history["accuracy"][np.argmax(history.history["val_accuracy"])])

            # updating the accuracy on the validation set
            val_accuracy += ((1/len(models)) * max(history.history["val_accuracy"]))
            
            # updating the predictions
            y_pred += ((1/len(models)) * models[n].predict(x=datasets["test"], verbose=0))

    return models, train_accuracy, val_accuracy, histories, y_pred



def plot_history_ensemble(histories, plot_size):
    """
    Plots the training and validation accuracy curves for each model in the ensemble.

    Args:
        histories (list): List of training history dictionaries for the ensemble members.
        plot_size (tuple): Size of the output plot in inches (width, height).

    Returns:
        None
    """
    # plotting the training histories of the ensemble
    plt.gcf()
    plt.figure(figsize=plot_size)
    for i in range(len(histories)):
        if i == 0:
            plt.plot(np.arange(1,len(histories[i]["accuracy"])+1), histories[i]["accuracy"], color="red", alpha=0.4, label="training accuracy")
            plt.plot(np.arange(1,len(histories[i]["val_accuracy"])+1), histories[i]["val_accuracy"], color="blue", alpha=0.4, label="validation accuracy")
        else:
            plt.plot(np.arange(1,len(histories[i]["accuracy"])+1), histories[i]["accuracy"], color="red", alpha=0.4)
            plt.plot(np.arange(1,len(histories[i]["val_accuracy"])+1), histories[i]["val_accuracy"], color="blue", alpha=0.4)
        
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("training history of the LeNet ensemble")
    plt.grid()
    plt.legend()