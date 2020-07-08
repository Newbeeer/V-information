import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from util_ import *
import numpy as np


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=0)
                                       for _ in range(nr_resnet)])
        self.uc_stream = nn.ModuleList([nn.Conv2d(nr_filters,nr_filters,(3,3),padding=1)
                                       for _ in range(nr_resnet)])
        self.uc_stream_bn = nn.ModuleList([nn.BatchNorm2d(nr_filters)
                                        for _ in range(nr_resnet)])

        # stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                                     resnet_nonlinearity, skip_connection=1)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, uc=None):
        u_list, ul_list, uc_list = [], [], []

        for i in range(self.nr_resnet):
            if uc is not None:
                uc = F.relu(self.uc_stream_bn[i](self.uc_stream[i](uc)))
                uc_list += [uc]
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u, c = uc)
            u_list += [u]
            ul_list += [ul]

        if uc is not None:
            return u_list, ul_list, uc_list
        else:
            return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=1)
                                       for _ in range(nr_resnet)])
        self.uc_stream = nn.ModuleList([nn.Conv2d(nr_filters, nr_filters, kernel_size=(3,3), padding=1)
                                       for _ in range(nr_resnet)])
        self.uc_stream_ = nn.ModuleList([nin(nr_filters, nr_filters) for _ in range(nr_resnet)])
        self.uc_stream_bn = nn.ModuleList([nn.BatchNorm2d(nr_filters)
                                        for _ in range(nr_resnet)])
        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                                     resnet_nonlinearity, skip_connection=2)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list, uc_list = None, uc = None):
        for i in range(self.nr_resnet):
            if uc_list is not None:
                uc = F.relu(self.uc_stream_bn[i](self.uc_stream[i](uc) + self.uc_stream_[i](uc_list.pop())))
            u = self.u_stream[i](u, a=u_list.pop(),c = uc)
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1), c =uc)
        if uc_list is not None:
            return u, ul, uc
        else:
            return u,ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                             self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                         self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in
                                                 range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in
                                               range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3),
                                          shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1, 3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2, 1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        #self.nin_out = nin(nr_filters, 1)
        self.nin_out = nin(nr_filters, 1)
        self.init_padding = None

    def forward(self, x, sample=False):
        # similar as done in the tf repo :
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)

        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1]) # + nr_resnet
            #print("i:", i, len(u_list), len(ul_list))
            u_list += u_out
            ul_list += ul_out
            #print("i:",i,len(u_list), len(ul_list))
            if i != 2:
                # downscale (only twice)

                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        #print(len(u_list),len(ul_list))
        ###    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            #print("i:", i, len(u_list), len(ul_list))
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)
            #print("i:", i, len(u_list), len(ul_list))
            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)



        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return F.sigmoid(x_out)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class PixelCNN_C(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN_C, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                             self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                         self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])
        self.downsize_uc_stream = nn.ModuleList([nn.Conv2d(nr_filters, nr_filters,kernel_size=(2,2),stride=(2, 2)) for _ in range(2)])
        self.downsize_uc_stream_bn = nn.ModuleList(
            [nn.BatchNorm2d(nr_filters) for _ in range(2)])
        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])
        self.upsize_uc_stream = nn.ModuleList([nn.ConvTranspose2d(nr_filters, nr_filters, kernel_size=(2,2), stride=(2,2)) for _ in range(2)])
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3),
                                          shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1, 3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2, 1), shift_output_right=True)])
        self.conv1 = nn.Conv2d(input_channels,nr_filters,(3,3),padding=1)
        self.bn1 = nn.BatchNorm2d(nr_filters)
        num_mix = 3 if self.input_channels == 1 else 10
        #self.nin_out = nin(nr_filters, 1)
        self.nin_out = nin(nr_filters, 1)
        self.init_padding = None
        self.act = nn.Sigmoid()


    def forward(self, x, x_, sample=False):
        # similar as done in the tf repo :
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        uc_list =  [F.relu(self.bn1(self.conv1(x_)))]
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            #print(i,":",u_list[-1].size(), ul_list[-1].size(),uc_list[-1].size())
            u_out, ul_out, uc_out = self.up_layers[i](u_list[-1], ul_list[-1],uc_list[-1]) # + nr_resnet

            u_list += u_out
            ul_list += ul_out
            uc_list += uc_out
            if i != 2:
                # downscale (only twice)

                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
                uc_list += [F.relu(self.downsize_uc_stream[i](uc_list[-1]))]
        #print(len(u_list),len(ul_list))
        ###    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()
        uc = uc_list.pop()
        for i in range(3):
            # resnet block
            #print("i:", i, len(u_list), len(ul_list))
            u, ul, uc = self.down_layers[i](u, ul,  u_list, ul_list, uc_list, uc)
            #print("i:", i, len(u_list), len(ul_list))
            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
                uc = F.relu(self.upsize_uc_stream[i](uc))


        x_out = self.nin_out(ul + uc)

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return self.act(x_out)


class PixelCNN_Cifar(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN_Cifar, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                             self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                         self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])
        self.downsize_uc_stream = nn.ModuleList([nn.Conv2d(nr_filters, nr_filters,kernel_size=(2,2),stride=(2, 2)) for _ in range(2)])
        self.downsize_uc_stream_bn = nn.ModuleList(
            [nn.BatchNorm2d(nr_filters) for _ in range(2)])
        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])
        self.upsize_uc_stream = nn.ModuleList([nn.ConvTranspose2d(nr_filters, nr_filters, kernel_size=(2,2), stride=(2,2)) for _ in range(2)])
        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3),
                                          shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1, 3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2, 1), shift_output_right=True)])
        self.conv1 = nn.Conv2d(input_channels,nr_filters,(3,3),padding=1)
        self.bn1 = nn.BatchNorm2d(nr_filters)
        num_mix = 3 if self.input_channels == 1 else 10
        #self.nin_out = nin(nr_filters, 1)
        self.nin_out = nin(nr_filters, input_channels)
        self.init_padding = None


    def forward(self, x, x_, sample=False):
        # similar as done in the tf repo :
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        uc_list =  [F.relu(self.bn1(self.conv1(x_)))]
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(2):
            # resnet block

            u_out, ul_out, uc_out = self.up_layers[i](u_list[-1], ul_list[-1],uc_list[-1]) # + nr_resnet

            u_list += u_out
            ul_list += ul_out
            uc_list += uc_out
            if i != 1:
                # downscale (only twice)

                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
                uc_list += [F.relu(self.downsize_uc_stream[i](uc_list[-1]))]
        #print(len(u_list),len(ul_list))
        ###    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()
        uc = uc_list.pop()
        for i in range(2):
            # resnet block

            u, ul, uc = self.down_layers[i](u, ul,  u_list, ul_list, uc_list, uc)

            if i != 1:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
                uc = F.relu(self.upsize_uc_stream[i](uc))


        x_out = self.nin_out(ul)

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out



if __name__ == '__main__':
    ''' testing loss with tf version '''
    np.random.seed(1)
    xx_t = (np.random.rand(15, 32, 32, 100) * 3).astype('float32')
    yy_t = np.random.uniform(-1, 1, size=(15, 32, 32, 3)).astype('float32')
    x_t = Variable(torch.from_numpy(xx_t)).cuda()
    y_t = Variable(torch.from_numpy(yy_t)).cuda()
    loss = discretized_mix_logistic_loss(y_t, x_t)

    ''' testing model and deconv dimensions '''
    x = torch.cuda.FloatTensor(32, 3, 32, 32).uniform_(-1., 1.)
    xv = Variable(x).cpu()
    ds = down_shifted_deconv2d(3, 40, stride=(2, 2))
    x_v = Variable(x)

    ''' testing loss compatibility '''
    model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=x.size(1))
    model = model.cuda()
    out = model(x_v)
    loss = discretized_mix_logistic_loss(x_v, out)
    print('loss : %s' % loss.data[0])