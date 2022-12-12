import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self):
        plt.ion()

        # self.fig, (self.ax1, self.ax2) = plt.subplots(1,2, figsize=(14,6))
        self.fig, self.ax1 = plt.subplots(1, figsize=(14,6))
        self.fig.suptitle('Learning Curves : Training')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.list_scores = []
        self.list_mean_scores = []
        self.list_mean_10_scores = []
        self.total_score = 0
        self.epoch = 0

    def update_lists(self, scores, epoch):
        self.epoch = epoch
        self.total_score += scores

        self.list_scores.append(scores)
        self.list_mean_scores.append(self.total_score / self.epoch)
        self.list_mean_10_scores.append(np.mean(self.list_scores[-10:]))

        self.update_plot()

    # def update_plot(self, scores, mean_scores, mean_10_scores, train_loss):
    def update_plot(self):
        
        self.ax1.clear()
        self.ax1.set_title('Scores :')
        self.ax1.set_xlabel('Number of Games')
        self.ax1.set_ylabel('score')
        self.ax1.plot(self.list_scores)
        self.ax1.plot(self.list_mean_scores)
        self.ax1.plot(self.list_mean_10_scores)
        self.ax1.set_ylim(ymin=0)
        self.ax1.text(len(self.list_scores)-1, self.list_scores[-1], str(self.list_scores[-1]))
        self.ax1.text(len(self.list_mean_scores)-1, self.list_mean_scores[-1], str(self.list_mean_scores[-1]))
        self.ax1.text(len(self.list_mean_10_scores)-1, self.list_mean_10_scores[-1], str(self.list_mean_10_scores[-1]))
        self.ax1.legend(['score', 'mean_score', 'tendancy'])
        # self.ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        # self.ax1.text(len(mean_10_scores)-1, mean_10_scores[-1], str(mean_10_scores[-1]))
        # self.ax1.legend(['score', 'mean_score', 'tendancy'])
        
        
        
        # self.ax2.clear()
        # self.ax2.set_title('Loss :')
        # self.ax2.set_xlabel('Number of Games')
        # self.ax2.set_ylabel('loss')
        # self.ax2.plot(train_loss)
        # self.ax2.set_ylim(ymin=0)
        # self.ax2.text(len(train_loss)-1, train_loss[-1], str(train_loss[-1]))
        # self.ax2.legend(['train_loss'])
        
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    
    