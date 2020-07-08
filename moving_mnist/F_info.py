import os
import time
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import seaborn as sns
from collections import defaultdict
from data import *
from models_conv import VAE
from models import F_MLP
plt.switch_backend('agg')

def main(args):
    ts = time.time()
    torch.cuda.set_device(args.device)


    def loss_fn(recon_x, x):


        return torch.sum((x-recon_x) ** 2) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(args.device)
    vae.load_state_dict(torch.load('model_conv2.pth.tar')['state_dict'])

    model = F_MLP(latent_size=args.latent_size).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)
    train_dataset_pair = Moving_MNIST_Frame(args.x, args.y)
    train_loader_pair = torch.utils.data.DataLoader(dataset=train_dataset_pair, batch_size=64, shuffle=True)

    for epoch in range(args.epochs):

        epoch_loss = 0.0
        for iteration, (x,y) in enumerate(train_loader_pair):

            x = x.to(args.device)
            x = x.unsqueeze(1)
            _, z_x, _, _ = vae(x)

            y = y.to(args.device)
            y = y.unsqueeze(1)
            _, z_y, _, _ = vae(y)

            mean_x = model(z_x)

            loss = loss_fn(mean_x, z_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data * x.size(0)

        print("{} -> {} :: Epoch : {}, Loss: {}".format(args.x,args.y,epoch,epoch_loss/len(train_loader.dataset)))




    #torch.save({'state_dict':vae.state_dict()},'model_conv2.pth.tar')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--rand", type=float, default=1.)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[4096, 512])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[512, 4096])
    parser.add_argument("--latent_size", type=int, default=10)
    parser.add_argument("--device", type=int)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true',default=False)

    args = parser.parse_args()

    main(args)