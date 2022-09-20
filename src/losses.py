import torch
from supn.losses.ipe_vae import kl_divergence_unit_normal
`
class L2VAELoss(torch.nn.Module):
    r"""
    L2 + KL loss with listener to log components into tensorboard.
    """

    def __init__(self, loss_logging_listener=None):
        r"""
        :loss_logging_listener: callable, used to log individual components of loss.
        """
        super().__init__()

        self.loss_logging_listener = loss_logging_listener
        self.l2_func = torch.nn.MSELoss()

    def forward(self, x, x_mu, z_mu, z_logvar):
        r"""
        Args:
            :x: (B, C, W, H) input image
            :x_mu: (B, C, W, H) output reconstruction mean
            :z_mu: (B, Cz) encoding mean
            :z_logvar: (B, Cz) encoding variance diagonal
        """
        # l2_
        l2_ = self.l2_func(x, x_mu)

        # KL divergence in latent space
        kl_ = kl_divergence_unit_normal(z_mu, z_logvar)

        ## Mean across batches [B,] -> float
        l2_ = torch.mean(nll_)
        kl_ = torch.mean(kl_)

        ## Log individual loss components if listener passed.
        if self.loss_logging_listener is not None:
            loss_dict = {
                    'l2' : l2_.item(),
                    'kl' : kl_.item()
                    }
            self.loss_logging_listener([], from_dict=loss_dict)

        return l2_ + kl_
