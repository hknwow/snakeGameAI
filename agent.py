import torch
import random
import numpy as np
from collections import deque
from snakeGame import SnakeGameAI, Direction, Point
from model import linearQNet, qTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
learninRate = 0.001

class Agent:
    
    def __init__(self):
        self.numGames = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = linearQNet(11, 256, 3) # input hidden output sizes
        self.trainer = qTrainer(self.model, learningRate=learninRate, gamma=self.gamma)

    def getState(self, game):
        head = game.snake[0]
        pointL = Point(head.x - 20, head.y)
        pointR = Point(head.x + 20, head.y)
        pointU = Point(head.x, head.y - 20)
        pointD = Point(head.x, head.y + 20)

        dirL = game.direction == Direction.LEFT
        dirR = game.direction == Direction.RIGHT
        dirU = game.direction == Direction.UP
        dirD = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dirR and game.is_collision(pointR)) or
            (dirL and game.is_collision(pointL)) or
            (dirU and game.is_collision(pointU)) or
            (dirD and game.is_collision(pointD)),

            # danger right
            (dirU and game.is_collision(pointR)) or
            (dirD and game.is_collision(pointL)) or
            (dirL and game.is_collision(pointU)) or
            (dirR and game.is_collision(pointD)),
            
            # danger left
            (dirU and game.is_collision(pointL)) or
            (dirD and game.is_collision(pointR)) or
            (dirL and game.is_collision(pointD)) or
            (dirR and game.is_collision(pointU)),

            # move direction
            dirL,
            dirR,
            dirU,
            dirD,

            # food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down (down direction y+)
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextState, gameOver):
        self.memory.append((state, action, reward, nextState, gameOver))
            # popleft if max_memory is reached

    def trainLongMemory(self): # replaying memory
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE)
        else:
            miniSample = self.memory
        
        states, actions, rewards, nextStates, gameOvers = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)

    def trainShortMemory(self, state, action, reward, nextState, gameOver):
        self.trainer.trainStep(state, action, reward, nextState, gameOver)

    def getAction(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.numGames # more games we have, the smaller epsilon will get
        finalMove = [0,0,0]
        if random.randint(0, 200) < self.epsilon: # and smaller the epsilon will get, the less frequent random
                                                # epsilon become negative and then we dont longer have a random move
            move = random.randint(0, 2)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # torch.argmax returns tensor .item() ile convert ediyorsun numbera
            finalMove[move] = 1
        
        return finalMove

def train():
    plotScore = []
    plotAvgScore = []
    totalScore = 0
    record = 0
    agent = Agent()
    # fileName='model.pth'
    # fileName = os.path.join('./model', fileName)
    # agent.model.load_state_dict(torch.load(fileName))
    game = SnakeGameAI()

    while True:
        # get the old state 
        stateOld = agent.getState(game)

        # get move
        finalMove = agent.getAction(stateOld)

        # perform move and get new state
        reward, gameOver, score = game.play_step(finalMove)
        stateNew = agent.getState(game)

        # train short memory
        agent.trainShortMemory(stateOld, finalMove, reward, stateNew, gameOver)

        # remember
        agent.remember(stateOld, finalMove, reward, stateNew, gameOver)

        if gameOver:
            # train long memory
            game.reset()
            agent.numGames += 1
            agent.trainLongMemory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.numGames, 'Score', score,
            'Record', record)
            
            plotScore.append(score)
            totalScore += score
            avgScore = totalScore / agent.numGames
            plotAvgScore.append(avgScore)
            plot(plotScore, plotAvgScore)

if __name__ == '__main__':
    train()