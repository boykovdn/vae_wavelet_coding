import torch
from supn.utils import rescale_to
from transforms import inverse_wavelet_transform
import numpy as np

def logging_wavelets_visualization(model, dset, inverse_transform, iteration, 
        summary_writer, device=0, n_imgs=1):
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

        :summary_writer: [previously torch.utils.tensorboard.SummaryWriter],
            now wendb instance (Weights and Biases).

        :device: int

        :n_imgs: int, the number of images to record.

    Returns:
        None

        Logs to tensorboard / wandb (the summary writer).
    """
    model.eval()

    if isinstance(dset[0], tuple):
        inp_shape = dset[0][0].shape # [C,H,W]
    else:
        inp_shape = dset[0].shape

    if n_imgs != 1:
        raise NotImplementedError()

    inputs = torch.zeros(n_imgs, *inp_shape)
    for idx in range(n_imgs):
        n_rand = int(torch.rand(1).item() * len(dset))
        inputs[idx] = dset[n_rand][0]

    inputs = inputs.to(device)
    with torch.no_grad():

        model_outp_ = model(inputs)
        assert isinstance(model_outp_, tuple), "Assuming SUPN model output format."
        out_mu = model_outp_[0]
        # Apply inverse wavelet transform to the output for visualization:
        img_i = inverse_wavelet_transform(out_mu, inverse_transform)
        img_i_target = inverse_wavelet_transform(inputs, inverse_transform)

    # TODO If we want to record multiple samples, this bit needs work.
    # Collect images here
    inputs_list = []
    outputs_list = []
    inv_input_list = []
    inv_output_list = []
    for idx_c in range(inputs.shape[1]):

        inputs_list.append(summary_writer.Image(inputs[0,idx_c]))
        outputs_list.append(summary_writer.Image(out_mu[0,idx_c]))
        
    inv_input_list.append(summary_writer.Image(
        img_i_target[0,0]
        )) # grayscale
    inv_output_list.append(summary_writer.Image(
        img_i[0,0]
        )) # grayscale

    # Log images here
    summary_writer.log({
                "Input wavelets" : inputs_list,
                "Output wavelets" : outputs_list,
                "Inverse input" : inv_input_list,
                "Inverse output" : inv_output_list,
                "Input highpass histogram" : summary_writer.Histogram(
                    np_histogram=np.histogram(inputs[0,-1].cpu())),
                "Output highpass histogram" : summary_writer.Histogram(
                    np_histogram=np.histogram(out_mu[0,-1].cpu())),
                "Output lowpass histogram" : summary_writer.Histogram(
                    np_histogram=np.histogram(out_mu[0,0].cpu()))
            })

    model.train()
