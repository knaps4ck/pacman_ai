from pacman import Directions
from game import Agent
import random
import util
import numpy as np


class IDSSearchAgent(Agent):

    def __init__(self, evalFn = 'evaluationFunction'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = 6

class IDSAgent(IDSSearchAgent):            

    def IDS(self, depth, gameState):                      
        if len(getLegalActionsWithNoStops(0, gameState)) == 0 or gameState.isLose() or depth == 0 or gameState.isWin():
            return self.evaluationFunction(gameState) - depth * 100
        currPos = gameState.getPacmanPosition()
        gameState.data.layout.walls[currPos[0]][currPos[1]] = True        
        val = []
        for max_depth in range(1, depth+1):        
            for action in getLegalActionsWithNoStops(0, gameState):
                val.append(self.IDS(max_depth - 1, gameState.generateSuccessor(0, action)))                
        max_val = max(val)                
        gameState.data.layout.walls[currPos[0]][currPos[1]] = False        
        return max_val + self.evaluationFunction(gameState) - depth * 100

    def getAction(self, gameState):       
        if gameState.getNumFood() <= self.depth:
            self.depth = gameState.getNumFood() - 1
        legalActions = getLegalActionsWithNoStops(0, gameState)
        action_scores = []
        currPos = gameState.getPacmanPosition()
        gameState.data.layout.walls[currPos[0]][currPos[1]] = True    
        for action in legalActions:            
            action_scores.append(self.IDS(self.depth, gameState.generateSuccessor(0, action)))
        gameState.data.layout.walls[currPos[0]][currPos[1]] = False
        max_action = max(action_scores)        
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)                
        return legalActions[chosenIndex]

def evaluationFunction(currState):
    currPos = currState.getPacmanPosition()
    currFood = currState.getFood()
    currGhostsPos = currState.getGhostStates()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in currGhostsPos]

    newFoodList = np.array(currFood.asList())
    foodDist = [util.euclideanDistance(currPos, food) for food in newFoodList]
    minFoodDistance = 0
    if len(newFoodList) > 0:
        minFoodDistance = foodDist[np.argmin(foodDist)]
        
    ghostPos = np.array(currState.getGhostPositions())   
    if len(ghostPos) > 0:
        ghostDist = [util.manhattanDistance(currPos, ghost) for ghost in ghostPos]
        minGhostDist = ghostDist[np.argmin(ghostDist)]
        nearestGhostScaredTime = ghostScaredTimes[np.argmin(ghostDist)]        
        if minGhostDist <= 1 and nearestGhostScaredTime == 0:
            return -999999        
        if minGhostDist <= 1 and nearestGhostScaredTime > 0:
            return 999999
    return currState.getScore() * 5 - minFoodDistance

    
def getLegalActionsWithNoStops(index, gameState):
    legalActions = gameState.getLegalActions(index)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    return legalActions