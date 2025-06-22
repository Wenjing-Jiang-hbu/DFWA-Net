import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_grayscale
from .pycontourlet.pycontourlet4d.pycontourlet import batch_multi_channel_pdfbdec

import time
def stack_same_dim(x):
    """Stack a list/dict of 4D tensors of same img dimension together."""
    # Collect tensor with same dimension into a dict of list
    output = {}

    # Input is list
    if isinstance(x, list):
        for i in range(len(x)):
            if isinstance(x[i], list):
                for j in range(len(x[i])):
                    shape = tuple(x[i][j].shape)
                    if shape in output.keys():
                        output[shape].append(x[i][j])
                    else:
                        output[shape] = [x[i][j]]
            else:
                shape = tuple(x[i].shape)
                if shape in output.keys():
                    output[shape].append(x[i])
                else:
                    output[shape] = [x[i]]
    else:
        for k in x.keys():
            shape = tuple(x[k].shape[2:4])
            if shape in output.keys():
                output[shape].append(x[k])
            else:
                output[shape] = [x[k]]

    # Concat the list of tensors into single tensor
    for k in output.keys():
        output[k] = torch.cat(output[k], dim=1)

    return output


class SSPCAB(nn.Module):


    def __init__(self, n_levs=[0, 3, 3, 3], variant="SSF", spec_type="all"):
        super(SSPCAB, self).__init__()

        # Model hyperparameters
        self.n_levs = n_levs
        self.variant = variant
        self.spec_type = spec_type

    def __pdfbdec(self, x, method="resize"):

        # if self.spec_type == 'avg':
        #     imgs = []
        #     # Iterate each image in a batch
        #     for i in range(x.shape[0]):
        #         # Convert to PIL and image and to grayscale image
        #         img = transforms.ToPILImage()(x[i])
        #         img = to_grayscale(img)
        #         imgs.append(img)
        #     # Restack and convert back to PyTorch tensor
        #     x = torch.from_numpy((np.expand_dims(np.stack(imgs, axis=0), axis=1)))

        # Obtain coefficients
        xlo, xhi = batch_multi_channel_pdfbdec(x=x.detach(), pfilt="maxflat", dfilt="dmaxflat7", nlevs=[0, 2],
                                            device=x.device)

        # # Stack channels with same image dimension
        # coefs = stack_same_dim(coefs)
        #
        # # Resize or splice
        # if method == "resize":
        #     for k in coefs.keys():
        #         # Resize if image is not square
        #         if k[2] != k[3]:
        #             # Get maximum dimension (height or width)
        #             max_dim = int(np.max((k[2], k[3])))
        #             # Resize the channels
        #             trans = transforms.Compose([transforms.Resize((max_dim, max_dim))])
        #             coefs[k] = trans(coefs[k])
        # else:
        #     for k in coefs.keys():
        #         # Resize if image is not square
        #         if k[2] != k[3]:
        #             # Get minimum dimension (height or width)
        #             min_dim = int(np.argmin((k[2], k[3]))) + 2
        #             # Splice alternate channels (always even number of channels exist)
        #             coefs[k] = torch.cat((coefs[k][:, ::2, :, :], coefs[k][:, 1::2, :, :]), dim=min_dim)

        # # Stack channels with same image dimension
        # coefs = stack_same_dim(coefs)
        #
        # # Change coefs's key to number (n-1 to 0), instead of dimension
        # for i, k in enumerate(coefs.copy()):
        #     idx = len(coefs.keys()) - i - 1
        #     coefs[idx] = coefs.pop(k)


        return xlo, xhi

    def forward(self, x):
        # shape = x.shape
        # if shape[2] < 640:
        #     down = False
        # else:
        #     down = True
        # if down:
        #     x_ = F.interpolate(x, size=(int(shape[2]/8), int(shape[3]/8)), mode='bilinear', align_corners=False)

        # if self.variant == "origin":
        #     xlo, xhi  = self.__pdfbdec(x, method="splice")
        # else:
        xlo, xhi  = self.__pdfbdec(x, method="resize")



        return xlo, xhi



def pad_4d_tensors(tensor1, tensor2):

    size1_3, size1_4 = tensor1.size(2), tensor1.size(3)
    size2_3, size2_4 = tensor2.size(2), tensor2.size(3)


    if size1_3 > size2_3:
        length = size1_3 - size2_3
        padding = (0, length, 0, length)
        tensor2 = torch.nn.functional.pad(tensor2, padding)

    if size1_3 < size2_3:
        tensor2 = tensor2[:,:,0:size1_3,0:size1_4]
    return tensor1, tensor2




class SSPCAT(nn.Module):

    def __init__(self, ichannel, ochannel, index):
        super(SSPCAT, self).__init__()

        self.conv = nn.Conv2d(ichannel, ochannel, kernel_size=1, stride=1, padding=0)
        self.index = index

    def forward(self, x):

        cf = x[1][0][self.index]
        f1, f2 = pad_4d_tensors(x[0],cf)
        f1 = f1.to(x[0].dtype)
        f2 = f2.to(x[0].dtype)
        output = self.conv(torch.cat((f1.to(x[0].device), f2.to(x[0].device)), 1))

        return output
