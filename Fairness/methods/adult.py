import torch
import torch.nn as nn
from adult_data import create_torch_dataloader
import argparse
from sklearn.metrics import roc_auc_score
from kernel_regression import KernelRegression
import numpy as np
class Encoder(nn.Module):

    def __init__(self,z_dim=10):

        super().__init__()


        self.MLP = nn.Sequential(
            nn.Linear(102,50),
            nn.Softplus()
        )
        self.linear_means = nn.Linear(50, z_dim)
        self.linear_log_var = nn.Linear(50, z_dim)

    def forward(self, x, u=None):

        #x = torch.cat((x,u),dim=1)
        x = self.MLP(x) # q(z | x, u)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, z_dim =10):

        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(z_dim,50),
        nn.Softplus(),
        nn.Linear(50,102),
        nn.Sigmoid())

    def forward(self, z, u=None):

        #z = torch.cat((z,u),dim=1)
        x = self.MLP(z)  # p(x | z, u)

        return x

class Logistic(nn.Module):

    def __init__(self):

        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(10,1),
                                 nn.Sigmoid())


    def forward(self, z):

        x = self.MLP(z)  # p(u | z)
        return x
class Discriminator(nn.Module):

    def __init__(self, z_dim =10, latent_size = 50, relu = False):

        super().__init__()
        if relu:
            self.MLP = nn.Sequential(nn.Linear(z_dim,latent_size),
            nn.ReLU(),
            nn.Linear(latent_size,1),
            nn.Sigmoid())
        else:
            self.MLP = nn.Sequential(nn.Linear(z_dim, latent_size),
                                     nn.Softplus(),
                                     nn.Linear(latent_size, 1),
                                     nn.Sigmoid())

    def forward(self, z):

        x = self.MLP(z)  # p(u | z)
        return x

class Discriminator_depth(nn.Module):

    def __init__(self, z_dim =10, latent_size = 100,relu=False):

        super().__init__()
        if relu:
            self.MLP = nn.Sequential(nn.Linear(z_dim, latent_size),
                                     nn.ReLU(),
                                     nn.Linear(latent_size, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, 1),
                                     nn.Sigmoid())
        else:
            self.MLP = nn.Sequential(nn.Linear(z_dim, latent_size),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(latent_size, 50),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(50, 1),
                                     nn.Sigmoid())


    def forward(self, z):

        x = self.MLP(z)  # p(u | z)
        return x

class Classifier(nn.Module):

    def __init__(self, z_dim =10):

        super().__init__()

        self.MLP = nn.Sequential(nn.Linear(z_dim, 50),
                                 nn.Softplus(),
                                 nn.Linear(50, 1),
                                 nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)  # p(y | z)
        return x

class VAE_x(nn.Module):

    def __init__(self,z_dim=10):

        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(z_dim=z_dim).cuda()
        self.decoder = Decoder(z_dim=z_dim).cuda()

    def forward(self, x,u, classifier=False):


        batch_size = x.size(0)
        means, log_var = self.encoder(x)
        if classifier:
            return means
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.z_dim]).cuda()
        z = eps * std + means

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).cuda()
        recon_x = self.decoder(z, c)

        return recon_x
class VAE(nn.Module):

    def __init__(self,z_dim=10):

        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(z_dim=z_dim).cuda()
        self.decoder = Decoder(z_dim=z_dim).cuda()

    def forward(self, x,u, classifier=False):


        batch_size = x.size(0)
        means, log_var = self.encoder(x, u)
        if classifier:
            return means
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.z_dim]).cuda()
        z = eps * std + means

        recon_x = self.decoder(z,u)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).cuda()
        recon_x = self.decoder(z, c)

        return recon_x

def loss_BCE(recon_x, x):

    BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, x.size(1)), x.view(-1, x.size(1)), size_average = False)

    return (BCE) / x.size(0)

def loss_KLD(mean, log_var):

    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD / mean.size(0)

def train_vae(vae_model,F,train_loader,test_loader,args,latent_size):


    e1 = 1
    e2 = 1
    e3 = 10
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))
    optimizer_F = torch.optim.Adam(F.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))
    for epoch in range(args.epochs):
        train_loss_v = 0.0
        train_loss_F = 0.0
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae_model(x,u)
            recon_u = F(mean)

            loss = e1 * loss_BCE(recon_x, x) + e2 * loss_KLD(mean, log_var) - e3 * loss_BCE(recon_u,u)
            #loss = e1 * loss_BCE(recon_x, x) + e2 * loss_KLD(mean, log_var)
            #loss = loss_BCE(recon_x,x)
            train_loss_v += loss.item()
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            for i  in range(1):
                recon_x, mean, log_var, z = vae_model(x, u)
                recon_u = F(mean)
                loss_F = loss_BCE(recon_u,u)
                train_loss_F += loss_F.item() * x.size(0)
                optimizer_F.zero_grad()
                loss_F.backward()
                optimizer_F.step()

        print("latent size : {}, epoch: {},  F loss : {}".format(latent_size,epoch,train_loss_F/len(train_loader.dataset)))

        if epoch % 50 == 0 and epoch!=0:
            torch.save(vae_model.state_dict(), 'vaex_model_relu_depth_adult_latent_'+str(latent_size)+str(epoch)+'.pth.tar')

        train_loss_F = 0.0
        correct = 0.0
        u_collect = []
        recon_u_collect = []
        for iteration, (x, u, y) in enumerate(test_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae_model(x, u)
            recon_u = F(mean)

            loss_F = loss_BCE(recon_u, u)
            train_loss_F += loss_F.item() * x.size(0)
            pred = (recon_u > 0.5).float()
            correct += (pred == u).float().sum()
            u_collect.append(u.detach().cpu())
            recon_u_collect.append(recon_u.detach().cpu())

        u = torch.cat(u_collect, dim=0).numpy()
        recon_u = torch.cat(recon_u_collect, dim=0).numpy()
        test_auc = roc_auc_score(u, recon_u)
        print("Test: latent size : {}, F information : {}, acc:{}, auc:{}".format(latent_size, 0.631475 - train_loss_F / len(test_loader.dataset), correct/len(test_loader.dataset),test_auc))

    torch.save(vae_model.state_dict(),'vaex_model_relu_depth_adult_latent_'+str(latent_size)+'.pth.tar')
    #torch.save(vae_model.state_dict(), 'vaex_model_adult_ori2.pth.tar')
    #torch.save(vae_model.state_dict(), 'vaex_model_depth.pth.tar')
    #torch.save(F.state_dict(), 'F_adult_latent_'+str(latent_size)+'.pth.tar')


def main(args):

    train_loader, test_loader = create_torch_dataloader(batch=64)
    vae = VAE_x(z_dim=10)
    F = Discriminator_depth(z_dim=10,latent_size=args.latent_size,relu=False).cuda()
    #3F = Discriminator(z_dim=10,latent_size=args.latent_size,relu=True).cuda()
    train_vae(vae,F,train_loader,test_loader,args,args.latent_size)
    #train_nearest_neighbor(vae, F, train_loader, test_loader, args, args.latent_size)
def train_classifier(args):
    train_loader, test_loader = create_torch_dataloader(batch=64)
    vae = VAE(z_dim=10)
    model_path = 'vae_model_adult.pth.tar'
    vae.load_state_dict(torch.load(model_path))
    #F = Discriminator(z_dim=10).cuda()
    classifier = Classifier(z_dim=10).cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        train_loss = 0.0
        tcorrect = 0.0
        correct = 0.0
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()
            mean = vae(x,u,classifier=True)
            output = classifier(mean)
            pre = (output > 0.5).detach().long()
            tcorrect += pre.eq(y).sum().item()
            loss = loss_BCE(output, y.float())
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        for iteration, (x, u, y) in enumerate(test_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda().long()
            mean = vae(x,u,classifier=True)
            output = classifier(mean)

            pre = (output > 0.5).detach().long()
            correct += pre.eq(y).sum().item()

        print("Epoch:{}, train acc : {}, test acc : {}".format(epoch,tcorrect/len(train_loader.dataset),correct/len(test_loader.dataset)))
    torch.save(classifier.state_dict(), 'classifier_adult.pth.tar')

def train_regression(model=None):
    def accuracy(y, y_logits):
        y_ = (y_logits > 0.0).astype(np.float32)
        return np.mean((y_ == y).astype(np.float32))
    train_loader, test_loader = create_torch_dataloader(batch=64)
    vae = VAE(z_dim=10)
    model_path = 'vae_model_adult_latent_100.pth.tar'
    #vae.load_state_dict(torch.load(model_path))
    if model != None:
        vae = model
    #F = Discriminator(z_dim=10).cuda()
    zs = []
    ys = []

    for iteration, (x, u, y) in enumerate(train_loader):
        x, u = x.cuda(), u.cuda()
        mean = vae(x, u, classifier=True)
        zs.append(mean.detach().cpu().numpy())
        ys.append(y.numpy())

    zs = np.concatenate(zs, axis=0)
    #print("Feature shape:",zs.shape)
    zsm = np.mean(zs, axis=0)
    zss = np.std(zs, axis=0)
    ys = np.concatenate(ys, axis=0)
    #print("Label shape:",ys.shape)
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit((zs - zsm) / zss, ys)
    ys_ = lr.predict((zs - zsm) / zss)
    from sklearn.metrics import roc_auc_score

    train_auc = roc_auc_score(ys, ys_)
    train_acc = accuracy(ys, ys_)

    zs = []
    ys = []
    for iteration, (x, u, y) in enumerate(test_loader):
        x, u = x.cuda(), u.cuda()
        mean = vae(x, u, classifier=True)
        zs.append(mean.detach().cpu().numpy())
        ys.append(y.numpy())

    zs = np.concatenate(zs, axis=0)
    #print("test Feature shape:", zs.shape)
    ys = np.concatenate(ys, axis=0)
    #print("test Label shape:", ys.shape)
    ys_ = lr.predict((zs - zsm) / zss)
    test_auc = roc_auc_score(ys, ys_),
    test_acc = accuracy(ys, ys_)

    print("train acc : {}, train auc : {}, test acc : {}, test auc : {}".format(train_acc,train_auc,test_acc,test_auc))
    return test_auc[0]

def F_informaiton_calculation(args,latent_size,model_latent_size):
    #Since H(u) is a constant, we can calculate it at the end of all experiments
    best_F = 100
    vae = VAE_x(z_dim=10)
    #model_path = 'vae_model_adult_latent_'+str(model_latent_size)+'.pth.tar'
    model_path = 'vaex_model_adult_kernel.pth.tar'
    #model_path = 'vaex_model_relu_adult_latent_50050.pth.tar'
    vae.load_state_dict(torch.load(model_path))
    #F = Discriminator(z_dim=10,latent_size=latent_size,relu=True).cuda()
    F = Logistic().cuda()
    #F = Discriminator_depth(z_dim=10, latent_size=latent_size, relu=False).cuda()
    optimizer_F = torch.optim.Adam(F.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    train_loader, test_loader = create_torch_dataloader(batch=64)
    train_loss_F = 0.0
    cnt = 0.
    best_auc = 0.0
    bw = 10
    for epoch in range(args.epochs):
        #if epoch == 200:
            #train_loss_F = 0.
            #cnt = 0.
        train_loss_F = 0.0
        correct = 0.0
        cnt = 0
        u_collect = []
        recon_u_collect = []
        for iteration, (x, u, y) in enumerate(train_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae(x, u)
            recon_u = F(z)

            loss_F = loss_BCE(recon_u, u)
            optimizer_F.zero_grad()
            loss_F.backward()
            optimizer_F.step()

        for iteration, (x, u, y) in enumerate(test_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae(x, u)
            recon_u = F(mean)
            #K = KernelRegression(bandwidth=bw, X=z, Y=u)
            #recon_u = K.predict_u(z)
            loss_F = loss_BCE(recon_u, u)
            train_loss_F += loss_F.item() * x.size(0)
            pred = (recon_u>0.5).float()
            correct += (pred == u).float().sum()
            u_collect.append(u.detach().cpu())
            recon_u_collect.append(recon_u.detach().cpu())

        u = torch.cat(u_collect, dim=0).numpy()
        recon_u = torch.cat(recon_u_collect, dim=0).numpy()
        test_auc = roc_auc_score(u, recon_u)
        if test_auc > best_auc:
            best_auc = test_auc
        print("epoch: {}, F loss : {}, acc: {}, auc: {}".format(epoch, 0.631475-train_loss_F/(len(test_loader.dataset) ), correct / len(test_loader.dataset), test_auc))
        if train_loss_F/(len(test_loader.dataset)) < best_F:
            best_F = train_loss_F/(len(test_loader.dataset))

    print("Model F={}, Latent size :{}, F informaiton(best) :{}".format(model_latent_size,latent_size,best_F))

    #torch.save(vae_model.state_dict(), 'vae_model_adult2.pth.tar')
    #torch.save(F.state_dict(), 'F_adult_'+str(latent_size)+'.pth.tar')

def test(args,latent_size):
    #Since H(u) is a constant, we can calculate it at the end of all experiments
    best_F = 100
    vae = VAE(z_dim=10)
    model_path = 'vae_model_adult_latent_50.pth.tar'
    vae.load_state_dict(torch.load(model_path))
    F = Discriminator(z_dim=10,latent_size=latent_size).cuda()
    model_path = 'F_adult_latent_50.pth.tar'
    F.load_state_dict(torch.load(model_path))

    train_loader, test_loader = create_torch_dataloader(batch=64)
    for epoch in range(1):
        train_loss_F = 0.0
        for iteration, (x, u, y) in enumerate(test_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae(x, u)
            recon_u = F(z)

            loss_F = loss_BCE(recon_u, u)
            train_loss_F += loss_F.item() * x.size(0)


        print("epoch: {},  H(u|z) loss : {}".format(epoch, best_F))

def Hz():

    train_loader, test_loader = create_torch_dataloader(batch=64)
    cnt = np.zeros((2))


    for iteration, (x, u, y) in enumerate(train_loader):

        cnt[0] += (u == 0).float().numpy().sum()
        cnt[1] += (u == 1).float().numpy().sum()

    cnt /= cnt.sum()
    hz = -(cnt[0] * np.log2(cnt[0]) + cnt[1] * np.log2(cnt[1]))
    cnt = torch.from_numpy(cnt[0].reshape(1,1))
    label = np.array([0]).reshape(1,1)
    label = torch.from_numpy(label).double()
    print(loss_BCE(cnt,label))
    print("H(z) = ",hz)

def train_nearest_neighbor(vae_model,F,train_loader,test_loader,args,latent_size):


    e1 = 1
    e2 = 1
    e3 = 100
    bw = 10
    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=args.learning_rate,betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        train_loss_v = 0.0
        z_collect = []
        u_collect = []
        for iteration, (x, u, y) in enumerate(train_loader):

            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae_model(x,u)
            K = KernelRegression(bandwidth=bw,X=z,Y=u)
            recon_u = K.predict_u(mean,train=True)

            loss = e1 * loss_BCE(recon_x, x) + e2 * loss_KLD(mean, log_var) - e3 * loss_BCE(recon_u,u)
            #loss = -e3 * loss_BCE(recon_u,u)
            train_loss_v += loss.item()
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            z_collect.append(mean)
            u_collect.append(u)

        z = torch.cat(z_collect, dim = 0)
        u = torch.cat(u_collect, dim = 0)
        K = KernelRegression(bandwidth=bw, X=z, Y=u)

        print("latent size : {}, epoch: {},  F loss : {}".format(latent_size,epoch,train_loss_v/len(train_loader.dataset)))
        train_loss_F = 0.0
        correct = 0.0
        u_collect = []
        recon_u_collect = []
        for iteration, (x, u, y) in enumerate(test_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae_model(x, u)
            recon_u = K.predict_u(mean)
            loss_F = loss_BCE(recon_u, u)
            train_loss_F += loss_F.item() * x.size(0)
            pred = (recon_u > 0.5).float()
            correct += (pred == u).float().sum()
            u_collect.append(u.detach().cpu())
            recon_u_collect.append(recon_u.detach().cpu())

        u = torch.cat(u_collect,dim=0).numpy()
        recon_u = torch.cat(recon_u_collect, dim=0).numpy()
        test_auc = roc_auc_score(u, recon_u)
        print("Test: latent size : {}, F information : {}, Acc : {}, Auc: {}".format(latent_size, 0.631475 - train_loss_F / len(test_loader.dataset), correct / len(test_loader.dataset),test_auc))

    torch.save(vae_model.state_dict(),'vaex_model_adult_kernel2.pth.tar')
    #torch.save(F.state_dict(), 'F_adult_latent_'+str(latent_size)+'.pth.tar')
def F_informaiton_calculation_kernel(args,latent_size,model_latent_size):
    #Since H(u) is a constant, we can calculate it at the end of all experiments
    best_F = 100
    vae = VAE_x(z_dim=10)
    #model_path = 'vae_model_adult_latent_'+str(model_latent_size)+'.pth.tar'
    model_path = 'vaex_model_adult_kernel.pth.tar'
    vae.load_state_dict(torch.load(model_path))
    train_loader, test_loader = create_torch_dataloader(batch=64)
    train_loss_F = 0.0
    cnt = 0.
    best_auc = 0.0
    bw = 15
    for epoch in range(args.epochs):
        bw = epoch+1
        train_loss_F = 0.0
        correct = 0.0
        cnt = 0
        u_collect = []
        z_collect = []

        for iteration, (x, u, y) in enumerate(train_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae(x, u)
            z_collect.append(mean)
            u_collect.append(u)

        z = torch.cat(z_collect, dim=0)
        u = torch.cat(u_collect, dim=0)
        K = KernelRegression(bandwidth=bw, X=z, Y=u)

        u_collect = []
        recon_u_collect = []
        for iteration, (x, u, y) in enumerate(test_loader):
            x, u, y = x.cuda(), u.cuda(), y.cuda()
            recon_x, mean, log_var, z = vae(x, u)

            recon_u = K.predict_u(mean)
            loss_F = loss_BCE(recon_u, u)
            train_loss_F += loss_F.item() * x.size(0)
            pred = (recon_u>0.5).float()
            correct += (pred == u).float().sum()
            u_collect.append(u.detach().cpu())
            recon_u_collect.append(recon_u.detach().cpu())

        u = torch.cat(u_collect, dim=0).numpy()
        recon_u = torch.cat(recon_u_collect, dim=0).numpy()
        test_auc = roc_auc_score(u, recon_u)
        if test_auc > best_auc:
            best_auc = test_auc
        print("epoch: {}, F loss : {}, acc: {}, auc: {}".format(epoch,0.631475- train_loss_F/(len(test_loader.dataset) ), correct / len(test_loader.dataset), test_auc))
        if train_loss_F/(len(test_loader.dataset)) < best_F:
            best_F = train_loss_F/(len(test_loader.dataset))

    print("Model F={}, Latent size :{}, F informaiton(best) :{}".format(model_latent_size,latent_size,best_F))

    #torch.save(vae_model.state_dict(), 'vae_model_adult2.pth.tar')
    #torch.save(F.state_dict(), 'F_adult_'+str(latent_size)+'.pth.tar')
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int)
    parser.add_argument("--ms", type=int)
    args = parser.parse_args()
    #torch.manual_seed(1234)
    #np.random.seed(1234)
    #main(args)
    #train_classifier(args)
    #train_regression()

    #F_informaiton_calculation(args,latent_size=args.latent_size,model_latent_size=args.ms)
    #test(args,args.latent_size)
    #Hz()
    #train_regression()
    #main(args)
    F_informaiton_calculation_kernel(args, latent_size=args.latent_size, model_latent_size=args.ms)