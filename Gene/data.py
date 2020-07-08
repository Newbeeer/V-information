import torch
import numpy as np
import torch.utils.data

class Two_Random(torch.utils.data.Dataset):


    def __init__(self, X, Y):

        self.X = X
        self.Y = Y


    def __getitem__(self, index):

        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


