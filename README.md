# SnakeGame-DeepQLearning-Pytorch

## Objective

### Main

* Finish the game completely

### Secondary : Skills acquired :

#### Python (Libraries):

* **Numpy**
* **Torch** (torch.nn, torch.optim, torch.multiprocessing)
* **matplotlib.pyplot**
* **tkinter**, pygame
* Collections (namedtuple, deque)
* Enum
* Time

#### General :

* **Parallel computing using multiprocessing** :heavy_check_mark:
* Reinforcement Learning : **Deep Q Learning** :heavy_check_mark:
* Reinforcement Learning : **Q Learning** :heavy_check_mark:
* Rendering graphics  :heavy_check_mark:
* Rendering plots  :heavy_check_mark:
* BFS (Breath First Search) :heavy_check_mark:
* Parallel computing using multithreading
* Compute on GPU

## Last version

* The branch "DQN+MultiEnv+MultiProcessing+Tkinter" if program using **multiprocessing** to run **4 environments**, **1 NN trainer**, **1 graphic displayer** in parallel. A complex structure with specific shared memory is used :

![structureprogram](https://user-images.githubusercontent.com/95492416/207930365-45df1074-a2ac-4897-8f6a-cc0c42e7e841.jpg)

* The graphic part is :
![v1Graphics](https://user-images.githubusercontent.com/95492416/207930043-bb5f076c-9453-42d3-ad39-f5298fbe3e77.png)

* The result is insane with a score of **11 in about 20 games** whereas the others which have in about 80 games (1st : efficient but slow) and 100 games (2nd (book) : bit less efficient but much faster.

## Diary

* day 30/11/2022 : DQN seem to be not efficient, after many tries, I was not able to improve it. En fact the snake was focus on eating the apple but when it started being long the random and learning part was over so he was not able to understand the recommanded direction (in the beginning this value was useless so the snake haven't improve this part)

* day 04/12/2022 : Algorith using BFS completely overtook the basic one (up to around 141)
I didn't expect QLearning would be better than the theorical algorith (only BFS) : (Deep) QLearning find strategy to deal with onlyBFS issues

* day 14/12/2022 : the network with the table as input finally works thanks to the version of DQN of the book which is more efficient. However it maybe cannot work for a single reason : DQN. Indeed, DQN algorithm just update the nn depending on one state (many but independantly) so with a given state, the algorith will try to find the best move. With this algorithm, nn canno't learn strategy, it can only learn a state and it's best action. Snake canno't be solved this way because there are 2*128^3 moves so the nn canno't learn perfectly all this moves.
Also we saw that a single hidden layer of 256n had an average score of 1.5 > 0.26 (random average) so it learnt but with this nn it canno't learn much more. Moreover with 2 hiddens layers of 512 had better result with an average of 16.5 > 1.5. Btw, on this grid 8x8, this is the best result I have but still far from 8x8 = 128
