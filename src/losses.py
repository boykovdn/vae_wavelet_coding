import torch
from supn.losses.ipe_vae import kl_divergence_unit_normal

class L2L1VAELoss(torch.nn.Module):
    r"""
    L2 + KL + L1 loss with listener to log components into tensorboard. The L2
    is appled in the low frequency channel, and the L1 loss on the rest.
    """

    def __init__(self, stdev=1., laplace_b=1., kl_weight=1.,
            loss_logging_listener=None):
        r"""
        :loss_logging_listener: callable, used to log individual components of loss.

        :laplace_b: float, the weighting factor for the L1 loss over the
            highpass filters. Equivalent to the b parameter in the Laplace
            distribution definition.

        :stdev: float, the standard deviation, used to scale the L2 loss.

        :kl_weight: float, the KL term weighting factor, default 1.
        """
        super().__init__()

        self.loss_logging_listener = loss_logging_listener
        self.l2_func = torch.nn.MSELoss(reduction='none')
        self.laplace_b = laplace_b
        self.kl_weight = kl_weight
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

        # L1 loss for high pass coefs, weighted by the laplace parameter b.
        # [B,C,H,W] -> [B,]
        l1_ = (x[:,1:] - x_mu[:,1:]).abs().sum((1,2,3)) / self.laplace_b 

        # KL divergence in latent space
        kl_ = self.kl_weight * kl_divergence_unit_normal(z_mu, z_logvar)

        ## Mean across batches [B,] -> float
        l2_ = torch.mean(l2_)
        kl_ = torch.mean(kl_)
        l1_ = torch.mean(l1_)

        ## Log individual loss components if listener passed.
        if self.loss_logging_listener is not None:

            loss_dict = {
                    'l2' : l2_.item(),
                    'kl' : kl_.item(),
                    'l1' : l1_.item()
                    }

            self.loss_logging_listener.log(loss_dict)

        return l2_ + kl_ + l1_
