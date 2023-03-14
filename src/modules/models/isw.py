# BSD 3-Clause License

# Copyright (c) 2021, Sungha Choi
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import kmeans1d
from .cov_attention import get_covariance_matrix, get_var_matrix

# reimplemented from https://github.com/shachoi/RobustNet/blob/main/network/cov_settings.py
class ISW(nn.Module):
    def __init__(self, clusters=50):
        super(ISW, self).__init__()
        self.clusters = clusters


    def get_mask_matrix(self, var_matrix):

        dim = var_matrix.shape[-1]
        var_flatten = torch.flatten(var_matrix)
        
        clusters, centroids = kmeans1d.cluster(var_flatten, self.clusters) # 50 clusters
        num_sensitive = var_flatten.size()[0] - clusters.count(0)  # 1: Insensitive Cov, 2~50: Sensitive Cov
        _, indices = torch.topk(var_flatten, k=int(num_sensitive))

        mask_matrix = torch.flatten(torch.zeros(dim, dim).cuda())
        mask_matrix[indices] = 1

        mask_matrix = mask_matrix.view(dim, dim)

        return mask_matrix


    def forward(self, x, x_tf):

        x = F.instance_norm(x)
        x_tf = F.instance_norm(x_tf)

        cov_base, _ = get_covariance_matrix(x)
        cov_tf, _ = get_covariance_matrix(x_tf)
    
        var_matrix = get_var_matrix(cov_base, cov_tf)
        var_matrix = torch.mean(var_matrix, dim=0, keepdim=True)

        mask = self.get_mask_matrix(torch.triu(var_matrix, diagonal=1))
        
        cov_base = cov_base * mask
        num_remove_cov = mask.sum() + 0.0001
        B = cov_base.shape[0]

        off_diag_sum = torch.sum(torch.abs(cov_base), dim=(1,2), keepdim=True) # - margin # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss
