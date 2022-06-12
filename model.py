import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class linearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, fileName='model.pth'):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        fileName = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), fileName)

class qTrainer:
    def __init__(self, model, learningRate, gamma):
        self.learningRate = learningRate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x) tek dim olduğu için tensor hale getiriyor gibi bişi
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )

        # simplified bellman equation
        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(gameOver)):
            qNew = reward[idx]
            if not gameOver[idx]:
                qNew = reward[idx] + self.gamma * torch.max(self.model(nextState[idx]))
            
            target[idx][torch.argmax(action[idx]).item()] = qNew

        # 2:qNew = r + y * max(next predicted Q value) -> only do this if not game over
        # pred.clone()
        # preds[argmax(action)] = qNew
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

