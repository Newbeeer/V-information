import torch
import numpy as np
from torchvision import datasets,transforms



class SaturationTrasform(object):
    '''
    for each pixel v: v' = sign(2v - 1) * |2v - 1|^{2/p}  * 0.5 + 0.5
    then clip -> (0, 1)
    '''

    def __init__(self, saturation_level = 2.0):
        self.p = saturation_level

    def __call__(self, img):

        ones = torch.ones_like(img)
        #print(img.size(), torch.max(img), torch.min(img))
        ret_img = torch.sign(2 * img - ones) * torch.pow( torch.abs(2 * img - ones), 2.0/self.p)

        ret_img =  ret_img * 0.5 + ones * 0.5

        ret_img = torch.clamp(ret_img,0,1)

        return ret_img


class Moving_MNIST(torch.utils.data.Dataset):


    def __init__(self, entropy=False, idx = None):
        dataset = np.load('/home1/xuyilun/moving_mnist/moving_sto.npy').astype(np.float32) / 255  # 20,10000,64,64

        dataset = (dataset > 0).astype(np.float32)

        if not entropy:
            self.img = dataset.reshape(-1,64,64)
        else:
            self.img = dataset[idx].reshape(-1,64,64)

    def __getitem__(self, index):

        return self.img[index]

    def __len__(self):
        return self.img.shape[0]

class Moving_MNIST_Frame(torch.utils.data.Dataset):


    def __init__(self, x,y):
        dataset = np.load('/home1/xuyilun/moving_mnist/moving_fix.npy').astype(np.float32) / 255  # 20,10000,64,64

        dataset = (dataset > 0).astype(np.float32)
        self.img_x = dataset[x].reshape(-1, 64, 64)
        self.img_y = dataset[y].reshape(-1, 64, 64)
    def __getitem__(self, index):

        return self.img_x[index],self.img_y[index]

    def __len__(self):
        return self.img_x.shape[0]



train_dataset = Moving_MNIST()
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)


root = '/home1/xuyilun/cifar'
batch_size = 64
train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ])
test_transform =  transforms.Compose([
    transforms.ToTensor(),

    ])
train_dataset_cifar = datasets.CIFAR10(root=root, train=True, transform=train_transform,download=False)
test_dataset_cifar = datasets.CIFAR10(root=root,train=False,transform=test_transform,download=False)

train_data_loader_cifar = torch.utils.data.DataLoader(dataset=train_dataset_cifar, batch_size=batch_size,num_workers = 16, shuffle=True)
test_data_loader_cifar = torch.utils.data.DataLoader(dataset=test_dataset_cifar, batch_size=batch_size,num_workers = 16, shuffle=True)
