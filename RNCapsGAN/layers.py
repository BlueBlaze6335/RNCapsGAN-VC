
import torch
import torch.nn as nn
import torch.nn.functional as F



class SquashHinton(nn.Module):
    def __init__(self, eps=10e-21):
        super(SquashHinton, self).__init__()
        self.eps = eps

    def forward(self, s):
        n = torch.norm(s, dim=-1, keepdim=True)
        return (n ** 2 / (1 + n ** 2) / (n + self.eps)) * s

class Squash(nn.Module):
    def __init__(self, eps=10e-21):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(self, s):
        n = torch.norm(s, dim=-1, keepdim=True)
        return (1 - 1 / (torch.exp(n) + self.eps)) * (s / (n + self.eps))

class PrimaryCaps(nn.Module):
    def __init__(self):
        super(PrimaryCaps, self).__init__()
        self.DW_Conv2D = nn.Conv2d(64,32, 4, stride=1)

    def forward(self, inputs):
        x = self.DW_Conv2D(inputs)
        x = x.view(-1, 64, 35)
        #print(x.shape, "view")
        x = Squash()(x)
        #print(x.shape ,"squash")
        return x

class FCCaps(nn.Module):
    def __init__(self, kernel_initializer='he_normal'):
        super(FCCaps, self).__init__()
        self.kernel_initializer = kernel_initializer
        self.W = nn.Parameter(torch.empty(32, 64, 35 ,17))
        self.b = nn.Parameter(torch.zeros( 1, 32,64, 1))

    def forward(self, inputs):
        #inputs_reshaped = inputs.unsqueeze(2).unsqueeze(-1)
        # Reshape self.W to [1, 1, 10, 16, 16] using unsqueeze
        #self.W_reshaped = self.W.unsqueeze(0).unsqueeze(1)
        u = torch.einsum('...ji,...kjiz->...kjz', inputs, self.W)
        c = torch.einsum('...ij,...kj->...i', u, u)[..., None]
        c = c / torch.sqrt(torch.tensor(16, dtype=torch.float32))
        #print(c.shape)
        c = F.softmax(c, dim=1)
        #print(c.shape)
        c = c + self.b
        print(c.shape, "c shape spftmax")
        s = torch.sum(u * c, dim=-2)
        v = Squash()(s)
        return v

class Length(nn.Module):
    def forward(self, inputs):
        return torch.sqrt(torch.sum(torch.square(inputs), -1) + torch.finfo(torch.float32).eps)

class Mask(nn.Module):
    def forward(self, inputs, double_mask=None):
        if isinstance(inputs, list):
            if double_mask:
                inputs, mask1, mask2 = inputs
            else:
                inputs, mask = inputs
        else:
            x = torch.sqrt(torch.sum(torch.square(inputs), -1))
            if double_mask:
                indices = torch.argsort(x, descending=True, dim=-1)[..., :2]
                mask1 = F.one_hot(indices[..., 0], num_classes=x.size(1))
                mask2 = F.one_hot(indices[..., 1], num_classes=x.size(1))
            else:
                indices = torch.argmax(x, dim=1)
                mask = F.one_hot(indices, num_classes=x.size(1))

        if double_mask:
            masked1 = inputs * mask1.unsqueeze(-1)
            masked2 = inputs * mask2.unsqueeze(-1)
            return masked1

class GLU(nn.Module):
    """Custom implementation of GLU since the paper assumes GLU won't reduce
    the dimension of tensor by 2.
    """

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class PixelShuffle(nn.Module):
    """Custom implementation pf Pixel Shuffle since PyTorch's PixelShuffle
    requires a 4D input (we have 3D inputs).
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        n = x.shape[0]
        c_out = x.shape[1] // 2
        w_new = x.shape[2] * 2
        return x.view(n, c_out, w_new)

class ConvLayer(nn.Module):

    def __init__(self, in_channels=2, out_channels=64):
        '''Constructs the ConvLayer with a specified input and output size.
           param in_channels: input depth of an image, default value = 1
           param out_channels: output depth of the convolutional layer, default value = 256
           '''
        super(ConvLayer, self).__init__()

        # defining a convolutional layer of the specified size
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=9, stride=1, padding=0)

    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input to the layer; an input image
           return: a relu-activated, convolutional layer
           '''
        # applying a ReLu activation to the outputs of the conv layer
        features = F.relu(self.conv(x))  # will have dimensions (batch_size, 20, 20, 256)
        return features


class ResidualLayer(nn.Module):
    """ResBlock.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, x):
        h1_norm = self.conv1d_layer(x)
        h1_gates_norm = self.conv_layer_gates(x)
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)  # GLU
        h2_norm = self.conv1d_out_layer(h1_glu)
        return x + h2_norm


class RN_B(nn.Module):
    def __init__(self, feature_channels):
        super(RN_B, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
               condition Mask: (B,1,H,W): 0 for background, 1 for foreground
        return: tensor RN_B(x): (N,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # RN
        #self.rn = RN_binarylabel(feature_channels)    # need no external parameters

        # gamma and beta
        self.foreground_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.bn_norm=nn.BatchNorm1d(feature_channels, affine=False, track_running_stats=False)
        #self.bn_norm=BN(affine=False)

    def forward(self, x, mask):
        # mask = F.adaptive_max_pool2d(mask, output_size=x.size()[2:])
        #mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')   # after down-sampling, there can be all-zero mask

        mask1=torch.zeros(size=x.shape)
        #mask.to("cuda:0")
        #print(f"Mask shape {mask.shape}")
        #print(f"Image shape {x.shape}")
        for i in range(min(mask.shape[0],x.shape[0])):
          for j in range(min(x.shape[1],mask.shape[1])):
            for k in range(min(x.shape[2],mask.shape[2])):
              mask1[i][j][k]=mask[i][j][k];
        #rn_x = self.rn(x, mask)\
        #print(f"Mask shape {mask.shape}")
        #mask1=mask1.to("cuda:0")
        rn_x_f=self.bn_norm(x*mask1)
        #print(f"RNf shape {rn_x_f.shape}")
        rn_x_b=self.bn_norm(x*(1-mask1))
        #print(f"RNb shape {rn_x_b.shape}")
        rn_x=rn_x_f+rn_x_b
        #print(f"rnx {rn_x.shape}")

        rn_x_foreground = (rn_x*mask1) * (1 + self.foreground_gamma[None,:,None]) + self.foreground_beta[None,:,None]
        rn_x_background = (rn_x*(1-mask1)) * (1 + self.background_gamma[None,:,None]) + self.background_beta[None,:,None]

        return rn_x_foreground + rn_x_background

class DownSampleGenerator(nn.Module):
    """Downsampling blocks of the Generator.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSampleGenerator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True))


        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True))



    def forward(self, x):
        # GLU
        return self.convLayer(x) * torch.sigmoid(self.convLayer_gates(x))



