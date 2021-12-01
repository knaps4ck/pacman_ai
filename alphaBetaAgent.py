from game import Directions
import random, util
import numpy as np

from game import Agent

class AlphaBetaSearchAgent(Agent):
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth='2'):
        self.index = 0 
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(AlphaBetaSearchAgent):
    def minimax(self, agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
                       getLegalActionsWithNoStops(0, gameState))
        else:
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
                       getLegalActionsWithNoStops(agent, gameState))

    def getAction(self, gameState):
        legalActions = getLegalActionsWithNoStops(0, gameState)
        action_scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action
                         in legalActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return legalActions[chosenIndex]

class AlphaBetaAgent(AlphaBetaSearchAgent):
    def alphabeta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:
            value = -999999
            for action in getLegalActionsWithNoStops(agent, gameState):
                value = max(value, self.alphabeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            for action in getLegalActionsWithNoStops(agent, gameState):
                value = 999999
                value = min(value, self.alphabeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def getAction(self, gameState):       
        legalActions = getLegalActionsWithNoStops(0, gameState)
        alpha = -999999
        beta = 999999
        action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), alpha, beta) for action
                         in legalActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return legalActions[chosenIndex]

def scoreEvaluationFunction(currState):
    return currState.getScore()

def evaluationFunction(currState, action):   
    successorGameState = currState.generatePacmanSuccessor(action)
    currPos = successorGameState.getPacmanPosition()
    currFood = successorGameState.getFood()
    currGhostsPos = successorGameState.getGhostStates()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in currGhostsPos]
   
    newFoodList = np.array(currFood.asList())
    distanceToFood = [util.manhattanDistance(currPos, food) for food in newFoodList]
    minFoodDistance = 0
    if len(newFoodList) > 0:
        minFoodDistance = distanceToFood[np.argmin(distanceToFood)]
   
    ghostPos = np.array(successorGameState.getGhostPositions())
    ghostDist = [util.manhattanDistance(currPos, ghost) for ghost in ghostPos]
    minGhostDistance = 0
    nearestGhostScaredTime = 0
    if len(ghostPos) > 0:
        minGhostDistance = ghostDist[np.argmin(ghostDist)]
        nearestGhostScaredTime = ghostScaredTimes[np.argmin(ghostDist)]       
        if minGhostDistance <= 1 and nearestGhostScaredTime == 0:
            return -999999      
        if minGhostDistance <= 1 and nearestGhostScaredTime > 0:
            return 999999

    value = successorGameState.getScore() - minFoodDistance
    if nearestGhostScaredTime > 0:       
        value -= minGhostDistance
    else:
        value += minGhostDistance
    return value

def getLegalActionsWithNoStops(index, gameState):
    legalActions = gameState.getLegalActions(index)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    return legalActions
