import torch

def wavelet_transform_reshape(img, dwtf_module):
    r"""
    This function does not expect a batched input, because it is expected to be
    used as a dataset transform function.

    Args:
        :img: torch.Tensor [C,H,W]

        :dwtf_module: torch.Module, function which computes the forward wavelet 
            transform. The level and other parameters of the transform are 
            specified in this module.
    """

    assert dwtf_module.J == 1, \
            "Only level 1 wavelet transform is currently supported."

    # TODO Generalize a bit, currently empirically seems to work?
    lowpass, highpass = dwtf_module(img.unsqueeze(0))

    # outp_.shape = [4,H,W]
    outp_ = torch.cat([lowpass, highpass[0][0]], dim=1)[0]

    return outp_

def inverse_wavelet_transform(wimg, dwti_module):
    r"""
    Applies the inverse wavelet transform, reshaping the input wavelet
    transformed image into the correct format. Assumes a specific shape for
    the input in terms of channels. Unlike the forward wavelet transform 
    defined in this file, the inverse expects a batch dimension, becausee it
    will be used on outputs from the model, which are batched.

    Args:
        :wimg: torch.Tensor [B,4,H,W], a level 1 wavelet transformed image.

        :dwti_module: torch.Module, discrete wavelet transform inverse, the 
            function which returns a pixel space image from a wavelet 
            representation.

    Returns:
        torch.Tensor [B,1,H,W], image in pixel space.
    """

    lowpass = wimg[:,0].unsqueeze(1) # (B,1,H,W)
    highpass = [ wimg[:,1:].unsqueeze(1) ] # [(B,1,3,H,W)], list of 1 element.

    img = dwti_module( (lowpass, highpass) )

    return img

class OddReLU(torch.nn.Module):
    r"""
    If -eps < x < eps then set x = 0. Else x = x - sign(x) * eps. This function
    is used to encourage sparsity in the neural network output when put in the 
    last layer. Implemented as the sum of two transformed ReLU activations.
    """

    def __init__(self, eps=1e-2, trainable=False, device='cpu'):
        r"""
        Args:
            :eps: float

            :trainable: bool, whether the threshold for setting to 0 should be
                a learnable parameter.

            :device: str identifier for host or gpu.
        """
        super().__init__()

        self.relu = torch.nn.ReLU()
        self.eps = torch.Tensor([eps]).to(device)

        if trainable:
            self.eps.requires_grad = True

    def forward(self, x):
        return self.relu(x - self.eps) - self.relu( - self.eps - x )
