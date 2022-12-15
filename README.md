# SnakeGame-DeepQLearning-Pytorch

## Objective

### Main

* Finish the game completely

### Secondary : Skills acquired :

#### Python (Libraries):

* **Numpy** :heavy_check_mark:
* **Torch** (torch.nn, torch.optim, torch.multiprocessing) :heavy_check_mark:
* **matplotlib.pyplot** :heavy_check_mark:
* **tkinter**, pygame :heavy_check_mark:
* Collections (namedtuple, deque) :heavy_check_mark:
* Enum :heavy_check_mark:
* Time :heavy_check_mark:

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

* The result is insane with a score of **11 in about 20 games** whereas the others which have in about 80 games (1st : efficient but slow) and 100 games (2nd (book) : bit less efficient but much faster. I used a 8x8 grid.

## Previous version

* *from book* means that I used the DQN of the book (see references at the end) and *from video* means that I used the DQN of the video (see references at the end)

* *basic state* means 11 inputs (4 directions, 4 foods, 3 dangers(straight, right, left)) and *bfs state* means 12 imputs (4 directions, 4 foods, 4 recommanded directions(directions where the number of cells in this direction is the higher)). **Computation time is higher but better result : score 149 > 60 for the same DQN. It still canno't finish the game so I gave up this method.**

### DQN from Book + Basic State + Random Snake Spawn (grid 8x8)
![Capture d’écran 2022-12-15 193341](https://user-images.githubusercontent.com/95492416/207940141-f39f59bb-c45f-437b-975e-6f8fcf545409.png)
**DQN from Book was much faster so I could try with the all grid as state, It's working but not enough for this imput (1 night training, 2 hidden layers of 512, 250k games and only 13 as result (too far from 8x8 = 64 so this neural network will not be enough to finish the game**
![tableinput-2x512hlayers-250kgames-bestresult](https://user-images.githubusercontent.com/95492416/207940595-b7a3883c-927c-42f4-9a9b-3489adc9af96.png)

### DQN from Video + BFS State + Random Snake Spawn (grid 26x20)
![bestRecordDQNv2](https://user-images.githubusercontent.com/95492416/207938551-bd8d9f2a-6d94-44d4-bb41-9e045036d9b3.png)

### DQN from Video + Basic State + Random Snake Spawn (grid 26x20)
![v0](https://user-images.githubusercontent.com/95492416/207938041-56bbdae7-3942-490c-ab6f-52594a0eb994.png)

### Q-Learning + BFS State + Random Snake Spawn (grid 26x20)
![QLearningBestV2](https://user-images.githubusercontent.com/95492416/207938422-ad1ddf03-e455-48b2-b4c3-041227a55582.png)

## Diary

* day 30/11/2022 : DQN seem to be not efficient, after many tries, I was not able to improve it. En fact the snake was focus on eating the apple but when it started being long the random and learning part was over so he was not able to understand the recommanded direction (in the beginning this value was useless so the snake haven't improve this part)

* day 04/12/2022 : Algorith using BFS completely overtook the basic one (up to around 141)
I didn't expect QLearning would be better than the theorical algorith (only BFS) : (Deep) QLearning find strategy to deal with onlyBFS issues

* day 14/12/2022 : the network with the table as input finally works thanks to the version of DQN of the book which is more efficient. However it maybe cannot work for a single reason : DQN. Indeed, DQN algorithm just update the nn depending on one state (many but independantly) so with a given state, the algorith will try to find the best move. With this algorithm, nn canno't learn strategy, it can only learn a state and it's best action. Snake canno't be solved this way because there are 2*128^3 moves so the nn canno't learn perfectly all this moves.
Also we saw that a single hidden layer of 256n had an average score of 1.5 > 0.26 (random average) so it learnt but with this nn it canno't learn much more. Moreover with 2 hiddens layers of 512 had better result with an average of 16.5 > 1.5. Btw, on this grid 8x8, this is the best result I have but still far from 8x8 = 128

* day 15/12/2022 : I implemented multiprocessing to my DQN, so I could implement multi agents, and I swich from Pygame to Tkinter.

## References

### First video
Video to make my first neural network on snake
https://www.youtube.com/watch?v=L8ypSXwyBds&ab_channel=freeCodeCamp.org

### Book
*Deep Reinforcement Learning Hands-on* from Maxim Lapan, Packt edition

![Capture d’écran 2022-12-15 194449](https://user-images.githubusercontent.com/95492416/207942173-a94942d1-07da-4dc6-bac3-d60d8ee6e209.png)

### Others

#### Help me completely understand DQN
https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b

#### Python libraries documentations
* https://pytorch.org/tutorials/
* official ducumentation of libraries
