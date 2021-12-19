# Pacman AI Implementation

This implementation is the extension of the Pacman project http://ai.berkeley.edu/project_overview.html.

![Pacman gif](http://ai.berkeley.edu/images/pacman_game.gif)

## Dependencies 
Python >= 3.9
Numpy >= 1.21.4

## Human
Play game by running the below command and using ASWD or arrow keys:
```
$ python pacman.py
```

## Alpha-Beta pruning algorithm(baseline AI)
- Running Alpha-Beta algorithm on a pacman game with minimax.
```
$ python pacman.py -p AlphaBetaAgent -l openClassic
```

## IDS algorithm
- IDS: Running IDS algorithm to find the best possible path with evaluation function.
```
$ python pacman.py -p IDSAgent -l mediumClassic
```

# Neural Network
- CNN: Running neural network on Pacman
```
$ python pacmanNN.py -p PacmanDQNAgent -n 4000 -x 3000 -l mediumClassic
```
-n is the total number of steps, -x is the training set.

Different types of mazes are available in layouts directory. To run above algorithms on layout specified in layouts directory replace -l mediumClassic to -l {layoutName}
