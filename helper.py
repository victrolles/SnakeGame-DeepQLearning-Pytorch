import matplotlib.pyplot as plt
from IPython import display

class Plot:
    def __init__(self):
        plt.ion()

        self.fig, self.ax1 = plt.subplots(1, figsize=(8,6)) #(self.ax1, self.ax2)
        self.fig.suptitle('Learning Curves : Training')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        
    def update_plot(self, scores, mean_scores, mean_10_scores):
        
        self.ax1.clear()
        self.ax1.set_title('Scores :')
        self.ax1.set_xlabel('Number of Games')
        self.ax1.set_ylabel('score')
        self.ax1.plot(scores)
        self.ax1.plot(mean_scores)
        self.ax1.plot(mean_10_scores)
        self.ax1.set_ylim(ymin=0)
        self.ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
        self.ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        self.ax1.text(len(mean_10_scores)-1, mean_10_scores[-1], str(mean_10_scores[-1]))
        self.ax1.legend(['score', 'mean_score', 'tendancy'])
        
        
        
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
  



# def plot(scores, mean_scores):
#     display.clear_output(wait=True)
#     display.display(plt.gcf())
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Number of Games')
#     plt.ylabel('Score')
#     plt.plot(scores)
#     plt.plot(mean_scores)
#     plt.ylim(ymin=0)
#     plt.text(len(scores)-1, scores[-1], str(scores[-1]))
#     plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
#     plt.show(block=False)
#     plt.pause(.1)

def plot(scores, mean_scores, mean_10_scores, train_loss):
    display.clear_output(wait=True)
    # display.display(plt.gcf())
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
    fig.suptitle('Learning Curves : Training')
    # plt.clf()
    ax1.set_title('Scores :')
    ax1.plot(scores)
    ax1.plot(mean_scores)
    ax1.plot(mean_10_scores)
    ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax1.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    ax1.text(len(mean_10_scores)-1, mean_10_scores[-1], str(mean_10_scores[-1]))
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('score')
    ax1.set_ylim(0)
    ax1.legend(['scores', 'mean_scores', 'tendance'])
    
    ax2.set_title('Loss :')
    ax2.plot(train_loss)
    ax2.text(len(train_loss)-1, train_loss[-1], str(train_loss[-1]))
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('train_loss')
    ax2.set_ylim(0)
    ax2.legend(['losses'])
    
    