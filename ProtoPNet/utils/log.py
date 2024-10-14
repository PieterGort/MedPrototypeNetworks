import os
import wandb
import sys

def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
        # Question: do we need to flush()
    return logger, f.close

def log_wandb(log_type, scores, global_step, num_classes):
    """
    Logs metrics to wandb.

    Args:
        log_type (str): The type of logging, e.g., 'train' or 'test'.
        scores (dict): A dictionary containing metric scores.
        global_step (int): The global step to associate with the logging event.
    """
    # # Prefix each key with the log type (train or test)
    # log_data = {f"{log_type}/{key}": value for key, value in scores.items()}
    log_data = {}
    for key, value in scores.items():
        if key == 'Confusion Matrix' and log_type == 'test':
            labels = [str(i) for i in range(num_classes)]
            wandb.log({f"{log_type}/{key}": value}, step=global_step)
        else:
            log_data[f"{log_type}/{key}"] = value

    # Log the data to wandb with the specified step
    if log_data:
        wandb.log(log_data, step=global_step)

class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    print(sys.executable)
    wandb.login()
    wandb.init(project='test_heatmap', settings=wandb.Settings(start_method='thread'))
    # Example usage

    # Create a confusion matrix plot
    plot = wandb.plot.confusion_matrix(probs=None, y_true=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], preds=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], class_names=[str(i) for i in range(5)])
    
    scores_example = {
        "Accuracy": 0.95,
        "Loss": 0.05,
        "Confusion Matrix": plot
    }
    
    log_wandb('test', scores_example, 1, 5)

if __name__ == "__main__":
    main()