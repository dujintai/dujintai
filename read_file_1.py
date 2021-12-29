import math
import numpy as np
import pandas as pd
import sqlite3
from pandas import Series, DataFrame
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('4.csv')
df = df.drop(columns='Unnamed: 0')
df_value = df.values
df_value_float = df_value.astype(np.float64)
df_torch = torch.from_numpy(df_value_float)
x_dataset = TensorDataset(df_torch)
train_loader = DataLoader(dataset=x_dataset, batch_size=5, shuffle=True)


class AutoEncoder(nn.Module):   #创建模型
    def __init__(self):                                                  #初始化
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(22283, 10000),
            nn.Tanh(),
            nn.Linear(10000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 100),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 1000),
            nn.Tanh(),
            nn.Linear(1000, 10000),
            nn.Tanh(),
            nn.Linear(10000, 22283),
            nn.Tanh(),
            nn.Sigmoid(),
        )
    def forward(self, x):                                  #forward也是一个已经定义过的函数，在模型真正赋值后开始运作
        encoded = self.encoder(x)                          #借用前面定义的函数（这里是encoder和decoder),对数据惊醒训练，最后返回需要的值
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()                                #创建网络
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
loss_func = nn.MSELoss()
exit_flag = False
for epoch in range(10):
    for step, (x) in enumerate(train_loader):
        # if exit_flag:
        #     print(autoencoder.state_dict())
        #     break
        val = torch.tensor([item.detach().numpy() for item in x])
        val1 = val.to(torch.float32)
        b_x = val1.view(-1, 22283)  # batch x, shape (batch, 28*28)让输入数据和Linear里面定义的一样，从tensor变成需要的
        b_y = val1.view(-1, 22283)
        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)

        # if math.isnan(loss.item()):
        #     exit_flag = True
        #     print(autoencoder.state_dict())
        #     break
        print(autoencoder.state_dict())
        print('loss:', loss)
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()