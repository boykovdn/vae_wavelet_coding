import torch
from supn.ipe_vae_blocks import (
        EncoderModule,
        ReparametrizationModule,
        DecoderModule)

class WaveletVAE(torch.nn.Module):
    r"""
    VAE architecture ehich works on wavelet representations of images. Reuses
    the SUPN encoder and decoder modules.
    """

    def __init__(self, input_shape, encoding_dim, depth=7, dim_h=None, 
            final_mu_activation=None, encoder_kernel_size=3, 
            init_num_channels=1, output_channels=1):
        r"""
        """
        super().__init__()

        assert dim_h is not None

        self.encoder = EncoderModule(input_shape, encoding_dim, dim_h, 
                depth=depth, kernel_size=encoder_kernel_size, 
                init_num_channels=init_num_channels)

        self.reparametrize = ReparametrizationModule()

        self.mu_decoder = DecoderModule(encoding_dim, dim_h, 
                output_ch=output_channels, depth=depth, 
                init_side=self.encoder.final_side,
                final_activation_creator=final_mu_activation,
                init_num_channels=init_num_channels)

    def forward(self, x):

        z_mu, z_logvar = self.encoder(x)
        z_sampled = self.reparametrize(z_mu, z_logvar.exp())
        x_mu = self.mu_decoder(z_sampled)

        return x_mu, z_mu, z_logvar
