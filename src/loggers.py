import torch
from supn.utils import rescale_to
from transforms import inverse_wavelet_transform

def logging_wavelets_visualization(model, dset, inverse_transform, iteration, 
        summary_writer, device=0, use_rescaling=False):
    r"""
    Log a sample of reconstructed wavelet parameters and the corresponding 
    decoded image into tensorboard.

    Args:

        :model: torch.Module

        :dset: torch.utils.data.Dataset object

        :inverse_transform: transform applied to the model output, which should
            result in a BCHW image. The result is recorded in tensorboard. This
            object is passed to another function which handles the input format 
            and returns the correct output.

        :iteration: int

        :summary_writer: torch.utils.tensorboard.SummaryWriter

        :device: int

        :use_rescaling: bool, if True then the model training is being done with
            rescaling transforms and this function will make sure that the model
            pre-transforms the input before passing it to the network, and then
            transforms it back so that it can be inverted and visualized.

    Returns:
        None

        Logs to tensorboard (the summary writer).
    """
    model.eval()
    if use_rescaling:
        # This is so that the model knows that the inputs are not scaled, and will
        # apply the scaling before and after working on the inputs. This way, the
        # output will have the correct relative scales for inverting and 
        # visualizing.
        model.use_rescaling = True

    if isinstance(dset[0], tuple):
        inp_shape = dset[0][0].shape # [C,H,W]
    else:
        inp_shape = dset[0].shape

    inputs = torch.zeros(5, *inp_shape)
    for idx in range(5):
        n_rand = int(torch.rand(1).item() * len(dset))
        inputs[idx] = dset[n_rand][0]

    inputs = inputs.to(device)
    with torch.no_grad():

        model_outp_ = model(inputs)
        assert isinstance(model_outp_, tuple), "Assuming SUPN model output format."
        out_mu = model_outp_[0]
        # Apply inverse wavelet transform to the output for visualization:
        img_i = inverse_wavelet_transform(out_mu, inverse_transform)

    for idx_c in range(inputs.shape[1]):
        summary_writer.add_images("Input ch {}".format(idx_c), 
                rescale_to(inputs[:,idx_c].unsqueeze(1), to=(0,1)), iteration, dataformats="NCHW")

    for idx_c in range(out_mu.shape[1]):
        summary_writer.add_images("SUPN Mean wavelet channel {}".format(idx_c),
                rescale_to(out_mu[:,idx_c].unsqueeze(1), to=(0,1)), iteration, dataformats="NCHW")

    summary_writer.add_images("Inverse wavelet of output mean:",
            rescale_to(img_i, to=(0,1)), iteration, dataformats="NCHW")

    if use_rescaling:
        model.use_rescaling = False

    model.train()
