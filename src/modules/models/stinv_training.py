import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Reverse autograd. Adapted from:  https://github.com/tadeephuy/GradientReversal/blob/master/gradient_reversal/functional.py
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        alpha = ctx.alpha
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


class domain_predictor(torch.nn.Module):
    """This class is used for predicting domain/stain information 

     Attributes:
        fc_input_features (int): N from B x N after flattening feature maps B x C x H x W
        n_centers (int): The number of domains or in this case n of entries in the stain matrix 

    """
    def __init__(self, fc_input_features, n_centers):
        """
        Args:
            fc_input_features (int): N from B x N after flattening feature maps B x C x H x W
            n_centers (int): The number of domains or in this case n of entries in the stain matrix
        """
        super(domain_predictor, self).__init__()
        # domain predictor
        self.fc_feat_in = fc_input_features
        self.n_centers = n_centers
        self.E = fc_input_features // 2		
        self.domain_embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
        self.linear = torch.nn.Linear(in_features=self.E, out_features=self.n_centers)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.E)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        out = self.domain_embedding(x)
        
        out = self.bn(out)
        out = self.dropout(out)
        out = self.relu(out)

        domain_prob = self.linear(out)

        return domain_prob


# from https://github.com/ilmaro8/HE_adversarial_CNN/blob/main/train/Training_script_domain_adversarial_regressor.py
class domain_predictor_original(torch.nn.Module):
    def __init__(self, fc_input_features, n_centers):
        super(domain_predictor_original, self).__init__()
        # domain predictor
        self.fc_feat_in = fc_input_features
        self.n_centers = n_centers
        self.E = fc_input_features // 2	
        self.domain_embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
        self.linear = torch.nn.Linear(in_features=self.E, out_features=self.n_centers)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.E)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        out = self.domain_embedding(x)
        out = self.dropout(out)
        domain_prob = self.linear(out)

        return domain_prob
