# this file contains generic functions and classes required to perform certain tasks

import pandas as pd
import numpy as np
import optuna
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# Custom callback to report validation loss to Optuna
class OptunaReportCallback(Callback):
    """
    Custom Keras callback to report validation accuracy to Optuna during training.

    This callback is designed to be used within an Optuna trial. After each epoch,
    it reports the current validation accuracy to the trial object and optionally
    prunes the trial if performance is not promising.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object used to report intermediate results
            and perform early stopping (pruning) if applicable.

    Methods:
        on_epoch_end(epoch, logs):
            Reports `val_accuracy` to Optuna at the end of each epoch and prunes
            the trial if `should_prune()` returns True.
    """
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None:
            self.trial.report(val_accuracy, step=epoch)
            # Optional pruning
            if self.trial.should_prune():
                raise optuna.TrialPruned()
            


def generate_submission_file(dataset, y_pred, file_name):
    """
    Generates a submission CSV file from model predictions for a test dataset.

    This function creates a DataFrame that maps each image index (starting from 1)
    to its predicted label, then saves the result as a CSV file in the format
    expected by many competition platforms (e.g., Kaggle).

    Args:
        dataset (np.ndarray): The test dataset of shape (N, H, W) or (N, D),
            used to determine the number of predictions.
        y_pred (np.ndarray): The prediction array of shape (N, C), where each row
            is a vector of class probabilities or logits. The predicted label is taken
            as the argmax of each row.
        file_name (str): The path to the output CSV file.

    Returns:
        None. A CSV file is written to disk with columns: ["ImageId", "Label"]

    Example:
        >>> generate_submission_file(X_test, model.predict(X_test), "submission.csv")
    """
    # initializing the output dataframe
    output_df = pd.DataFrame(data=np.zeros(shape=(dataset.shape[0],2), dtype=np.int64), columns=["ImageId", "Label"])
    
    # predicting the classes of the images
    for i in range(dataset.shape[0]):
        output_df.loc[i,"ImageId"] = i + 1
        output_df.loc[i, "Label"] = np.argmax(y_pred[i,:])
        
    # saving the dataframe as a .csv file
    output_df.to_csv(file_name, index=False)