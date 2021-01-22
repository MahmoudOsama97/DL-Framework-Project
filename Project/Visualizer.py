import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    
    # This function is called when the training begins
    def __init__(self, mode= 'all'):
        # Initialize the lists for holding the logs, losses and metrics
        self.losses = []
        self.accuracy = []
        self.f1score = []
        self.precision = []
        self.recall = []
        self.logs = []
        self.mode = mode
        self.metric_mode = []
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        """
        Calculates and plots Precision, Recall, F1 score
        """
        # Extract from the log
        if self.mode == "all":
            accuracy = logs.get('accuracy', default= -1) 
            f1score = logs.get('f1score', default= -1) 
            recall = logs.get('recall', default= -1) 
            precision = logs.get('precision', default= -1)
            self.accuracy.append(accuracy)
            self.f1score.append(f1score)
            self.precision.append(precision)
            self.recall.append(recall)
        else:
            metric_mode = logs.get(self.mode)
            self.metric_mode.append(metric_mode)
        loss = logs.get('loss', default= -1)
        self.losses.append(loss)
    
        # Clear the previous plot
        plt.cla()
        N = np.arange(0, len(self.losses))
        
        # You can chose the style of your preference
        plt.style.use("seaborn")
        
        # Plot train loss, train acc, val loss and val acc against epochs passed
        plt.figure(figsize=(10,3))
        plt.title("Loss over epoch")
        plt.plot(N, self.losses)
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax = ax.ravel()
        if (self.mode == "all"):
            ax[0].plot(N, self.precision, label = "Precision", c = 'red')
            ax[1].plot(N, self.recall, label = "Recall", c = 'red')
            ax[2].plot(N, self.f1score, label = "F1 score", c = 'red')
            ax[3].plot(N, self.accuracy, label = "Precision", c = 'red')
            ax[0].set_title("Precision at Epoch No. {}".format(len(self.losses)))
            ax[1].set_title("Recall at Epoch No. {}".format(len(self.losses)))
            ax[2].set_title("F1-score at Epoch No. {}".format(len(self.losses)))
            ax[3].set_title("Accuracy at Epoch No. {}".format(len(self.losses)))
            ax[0].set_xlabel("Epoch #")
            ax[1].set_xlabel("Epoch #")
            ax[2].set_xlabel("Epoch #")
            ax[3].set_xlabel("Epoch #")
            ax[0].set_ylabel("Precision")
            ax[1].set_ylabel("Recall")
            ax[2].set_ylabel("F1 score")
            ax[3].set_ylabel("Accuracy")
            ax[0].set_ylim(0,1)
            ax[1].set_ylim(0,1)
            ax[2].set_ylim(0,1)
            ax[3].set_ylim(0,1)
        else:
            ax[0].plot(N, self.metric_mode, label = self.mode, c = 'red')
            ax[0].set_title("{} at Epoch No. {}".format(self.mode, len(self.losses)))
            ax[0].set_xlabel("Epoch #")
            ax[0].set_ylabel(self.mode)
            ax[0].set_ylim(0,1)
        plt.show()