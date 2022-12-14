# SnakeGame-DeepQLearning-Pytorch

First Image Deep Learning

![my ai](https://user-images.githubusercontent.com/95492416/203067541-d7d910f4-250f-4443-97aa-ee06e56cff2c.png)

First Video Deep Learning

Cannot insert it :you can find it in the Folder FeedBack-Data

day 30/11/2022 : DQN seem to be not efficient, after many tries, I was not able to improve it. En fact the snake was focus on eating the apple but when it started being long the random and learning part was over so he was not able to understand the recommanded direction (in the beginning this value was useless so the snake haven't improve this part)

day 14/14/2022 : the network with the table as input finally works thanks to the version of DQN of the book which is more efficient. However it maybe cannot work for a single reason : DQN. Indeed, DQN algorithm just update the nn depending on one state (many but independantly) so with a given state, the algorith will try to find the best move. With this algorithm, nn canno't learn strategy, it can only learn a state and it's best action. Snake canno't be solved this way because there are 2*128^3 moves so the nn canno't learn perfectly all this moves.
Also we saw that a single hidden layer of 256n had an average score of 1.5 > 0.26 (random average) so it learnt but with this nn it canno't learn much more. Moreover with 2 hiddens layers of 512 had better result with an average of 16.5 > 1.5. Btw, on this grid 8x8, this is the best result I have but still far from 8x8 = 128
