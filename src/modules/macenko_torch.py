import torch
from torchstain.base.normalizers.he_normalizer import HENormalizer
from torchstain.torch.utils import cov, percentile

"""
Source code ported from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization

Torch version is from: https://github.com/EIDOSLAB/torchstain
"""

# MIT License

# Copyright (c) 2020 EIDOSlab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class TorchMacenkoNormalizer(HENormalizer):
    """This class is used for applying stain normalization to histological images  
    """
    def __init__(self):
        super().__init__()

        self.HERef = torch.tensor([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io, beta):
        I = I.permute(1, 2, 0)

        # calculate optical density
        OD = -torch.log((I.reshape((-1, I.shape[-1])).float() + 1)/Io)

        # remove transparent pixels
        ODhat = OD[~torch.any(OD < beta, dim=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = torch.matmul(ODhat, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])

        minPhi = percentile(phi, alpha)
        maxPhi = percentile(phi, 100 - alpha)

        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = OD.T

        # determine concentrations of the individual stains
        return torch.lstsq(Y, HE)[0][:2]

    def __compute_matrices(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = torch.symeig(cov(ODhat.T), eigenvectors=True)
        eigvecs = eigvecs[:, [1, 2]]

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)
        maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def get_stain_matrix(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        ''' Finds optimal stain vectors of the given image

        Args:
            I (torch.tensor): RGB input image shape [C, H, W] and type uint8
            Io (int, optional) transmitted light intensity
            alpha (float, optional): percentile
            beta (float, optional): transparency threshold

        Returns:
            torch.tensor: Matrix containing optimal stain vectors
            torch.tensor: Concentrations of the individual stains
            torch.tensor: Maximum concentrations

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009

        '''

        stain_matrix, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        return stain_matrix, C, maxC


    def normalize(self, I, maxCRef, HERef, Io=240, alpha=1, beta=0.15, stains=True):
        ''' Normalize staining appearence of H&E stained images

        Example use:
            see test.py

        Input:
            I: RGB input image: tensor of shape [C, H, W] and type uint8
            Io: (optional) transmitted light intensity
            alpha: percentile
            beta: transparency threshold
            stains: if true, return also H & E components

        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        # normalize stain concentrations
        C *= (maxCRef / maxC).unsqueeze(-1)

        # recreate the image using reference mixing matrix
        Inorm = Io * torch.exp(-torch.matmul(HERef, C))
        Inorm[Inorm > 255] = 255
        Inorm = Inorm.T.reshape(h, w, c).int()

        H, E = None, None

        if stains:
            H = torch.mul(Io, torch.exp(torch.matmul(-HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))))
            H[H > 255] = 255
            H = H.T.reshape(h, w, c).int()

            E = torch.mul(Io, torch.exp(torch.matmul(-HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))))
            E[E > 255] = 255
            E = E.T.reshape(h, w, c).int()

        return Inorm, H, E
