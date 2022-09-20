import torch
from supn.utils import rescale_to

def logging_wavelets_visualization(model, dset, iteration, summary_writer,
        device=0):
    r"""
    Log a sample of reconstructed wavelet parameters and the corresponding 
    decoded image into tensorboard.

    Args:

        :model: torch.Module

        :dset: torch.utils.data.Dataset object

        :iteration: int

        :summary_writer: torch.utils.tensorboard.SummaryWriter

        :device: int

    Returns:
        None

        Logs to tensorboard (the summary writer).
    """
    model.eval()

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
        assert len(model_outp_) == 4, "Assuming SUPN model output format."
        out_mu = model_outp_[0]

    summary_writer.add_images("Input", 
            rescale_to(inputs, to=(0,1)), iteration, dataformats="NCHW")

    for idx_c in range(out_mu.shape[1]):
        summary_writer.add_images("SUPN Mean wavelet channel {}".format(idx_c),
                rescale_to(out_mu[:,idx_c].unsqueeze(1), to=(0,1)), iteration, dataformats="NCHW")

    model.train()
