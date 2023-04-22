import torch
import torch.nn as nn
import torch.nn.functional as F


class CovarianceAttention(nn.Module):
    """Used for computing channel attention based on feature covariances

    Attributes:
        in_channels (int): The number of input channels (C) from B x C x H x W
    """

    def __init__(self, in_channels):
        """
        Args:
            in_channels (int): The number of input channels (C) from B x C x H x W
        """
        super(CovarianceAttention, self).__init__()
        self.fc = nn.Linear(in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_tf):
        """Copmuter channel attention scores from feature covariances
        Steps:
        1) Compute individiaul covariance matrices (COV, COV') of feature maps of the input image (F) and its augmented version (F')
        2) Compute variance matrix (V) that represents difference between two covariance matrices (COV, COV')
        3) Apply linear transformation (FC) to get channel weights
        4) Map these values to the range (0, 1) with sigmoid

        Args:
            x (torch.tensor): Feature maps of the input image after n-th encoder stage, n is optional
            x_tf (torch.tensor): Feature maps of the transformed/augmendted input image after n-th encoder stage, n is optional

        Returns:
            torch.tensor: Channel weights
        """

        cov_base, _ = get_covariance_matrix(x)
        cov_tf, _ = get_covariance_matrix(x_tf)

        var_matrix = get_var_matrix(cov_base, cov_tf)

        channel_attention = self.fc(var_matrix).unsqueeze(3)
        channel_attention = self.sigmoid(channel_attention)

        return channel_attention


# https://github.com/shachoi/RobustNet/blob/main/network/instance_whitening.py
def get_covariance_matrix(f_map, eye=None):
    """Computes covariance matrix of the given feature maps

    Args:
        f_map (torch.tensor): Input feature maps
        eye (torch.tensor, optional): Identity matrix for denominator stabilization. Defaults to None.

    Returns:
        torch.tensor: Covariance matrix
        int: Batch size
    """
    eps = 1e-5
    f_map = f_map.to(torch.float32)
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (
        eps * eye
    )  # C X C / HW

    return f_cor, B


def get_var_matrix(cov, cov_tf):
    """Computes variance matrix from two covariance matrices

    Args:
        cov (torch.tensor): Covariance matrix of feature maps of the input image
        cov_tf (torch.tensor): Covariance matrix of feature maps of the transformed/augmented input image

    Returns:
        torch.tensor: Variance matrix
    """
    mu = 0.5 * (cov + cov_tf)
    sigma_sq = 0.5 * ((cov - mu) ** 2 + (cov_tf - mu) ** 2)

    return sigma_sq
