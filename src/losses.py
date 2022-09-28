import torch
from supn.losses.ipe_vae import kl_divergence_unit_normal

class L2VAELoss(torch.nn.Module):
    r"""
    L2 + KL loss with listener to log components into tensorboard.
    """

    def __init__(self, stdev=1., l1_weight=0., loss_logging_listener=None):
        r"""
        :loss_logging_listener: callable, used to log individual components of loss.

        :l1_weight: float, the weighting factor for L1 lasso regularization over
            the highpass channels.

        :stdev: float, the standard deviation, used to scale the L2 loss.
        """
        super().__init__()

        self.loss_logging_listener = loss_logging_listener
        self.l2_func = torch.nn.MSELoss(reduction='none')
        self.l1_weight = l1_weight
        self.stdev = stdev

    def forward(self, x, x_mu, z_mu, z_logvar):
        r"""
        Args:
            :x: (B, C, W, H) input image
            :x_mu: (B, C, W, H) output reconstruction mean
            :z_mu: (B, Cz) encoding mean
            :z_logvar: (B, Cz) encoding variance diagonal
        """
        # l2_ weighted by the variance
        l2_ = self.l2_func(x[:, 0:1], x_mu[:, 0:1]) / self.stdev**2

        # L1 loss for high pass coefs
        l1_ = self.l1_weight * (x[:,1:] - x_mu[:,1:]).abs().sum((1,2,3)) # [B,C,H,W] -> [B,]

        # KL divergence in latent space
        kl_ = kl_divergence_unit_normal(z_mu, z_logvar)

        ## Mean across batches [B,] -> float
        l2_ = torch.mean(l2_)
        kl_ = torch.mean(kl_)
        l1_ = torch.mean(l1_)

        ## Log individual loss components if listener passed.
        if self.loss_logging_listener is not None:
            ch_stds = x.std((0,2,3))
            loss_dict = {
                    'l2' : l2_.item(),
                    'kl' : kl_.item(),
                    'l1' : l1_.item(),
                    'std_ll' : ch_stds[0],
                    'std_lh' : ch_stds[1],
                    'std_hl' : ch_stds[2],
                    'std_hh' : ch_stds[3]
                    }
            self.loss_logging_listener([], from_dict=loss_dict)

        return l2_ + kl_ + l1_
