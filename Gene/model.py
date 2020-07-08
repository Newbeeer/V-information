import torch.nn as nn
import torch
import torch.nn.functional as F


class mlp(nn.Module):
    """
    the common architecture for the left model
    """
    def __init__(self,dim):
        super(mlp, self).__init__()

        latent_dim = 200
        self.fc1 = nn.Linear(1,latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(latent_dim,dim)
        self.bn2 = nn.BatchNorm1d(dim)
    def forward(self, x):

        x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class mlp_try(nn.Module):
    """
    the common architecture for the left model
    """
    def __init__(self,dim):
        super(mlp_try, self).__init__()


        self.fc1 = nn.Linear(3,1)

    def forward(self, x):

        x = self.fc1(x)

        return x