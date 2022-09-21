from supn.logging.ipe_vae import LoggingScalarListener
from supn.losses.ipe_vae import NegativeELBOLoss
from losses import L2VAELoss
import torch
from tqdm import tqdm

class IPEVAETrainer:
    r"""
    Class that handles the training of the IPE VAE. Uses the Adam optimizer and
    negative ELBO loss by default. Supports some basic logging and tries to 
    expose very few parameters.
    """

    def __init__(self, model, 
            training_type="vae_full", 
            summary_writer=None,
            learning_rate=1e-3,
            enforce_l2=False,
            stdev=None):
        r"""
        Args:

            :model: torch.Module, the IPE VAE.

            :training_type: str, selects which parts of the model will be
                passed to the optimizer, the rest will accumulate gradients,
                but not be updated (can also freeze to avoid calculating grad).

            :summary_writer: torch.utils.tensorboard.SummaryWriter object, used
                to log loss values and image reconstructions optionally. It is
                passed to the loss function, which handles the io to the logger
                separately.

            :learning_rate: float

            :enforce_l2: bool, if True will set the network output to be the
                identity, so that the loss amounts to L2 + KL.

            :stdev: float or None, if enforce_l2 is True, then we can further
                weigh the KL and L2 terms. This parameter is the standard 
                deviation, and will enter as a 1/stdev multiplier for the fake
                Cholesky decomposition.
        """
        self.training_type_options = ["vae_full", "chol_only", "mean_only"]
        
        self.training_type = training_type
        self.model = model
        self.learning_rate = learning_rate
        self.stdev = stdev

        self.enforce_l2 = enforce_l2

        if summary_writer is not None:
            listener = LoggingScalarListener(summary_writer)
        else:
            listener = None

        self.elbo_loss = NegativeELBOLoss(loss_logging_listener = listener)

    def create_new_optimizer(self):
        r"""
        Create the Adam optimizer in a separate function, because we might need
        to do this if train is called multiple times - a single optimizer would
        store history (such as momentum) between different training sessions.

        Manually sets the requires_grad flags of the relevant parts of the
        model.

        Returns:
            torch.optim.Adam optimizer.
        """

        if self.training_type == "mean_only":

            self.model.freeze( [self.model.var_decoder] )
            self.model.unfreeze( [self.model.encoder, self.model.mu_decoder] )

            optimizer = torch.optim.Adam([
                {'params' : self.model.encoder.parameters()},
                {'params' : self.model.mu_decoder.parameters()}
                ], lr=self.learning_rate)

        elif self.training_type == "chol_only":

            self.model.freeze( [self.model.encoder, self.model.mu_decoder] )
            self.model.unfreeze( [self.model.var_decoder] )

            optimizer = torch.optim.Adam([
                {'params' : self.model.var_decoder.parameters()}
                ], lr=self.learning_rate)

        elif self.training_type == "vae_full":

            self.model.unfreeze( [self.model.encoder, self.model.mu_decoder, self.model.var_decoder] )

            optimizer = torch.optim.Adam(
                    self.model.parameters(), 
                    lr=self.learning_rate)

        else:
            raise Exception(
                    "Invalid training type {}".format(self.training_type)
                    )

        return optimizer

    def load_state(self, state_path):
        r"""
        Load the state dict into the model.

        Args:
            :state_path: Path
        """
        state_dict = torch.load(state_path)
        self.model.load_state_dict(state_dict)

    def save_state(self, state_path):
        r"""
        Save the current model's state dict.

        Args:
            :state_path: Path
        """
        torch.save(self.model.state_dict(), state_path)

    def train(self, dataloader, save_state_path, 
            image_logging=None, 
            device=0, max_iterations=10000, 
            init_state_path=None):
        r"""
        Runs the training loop and saves the final state dict.

        Args:

            :dataloader: torch.utils.data.DataLoader object which wraps the
                training dataset.

            :save_state_path: path or str, output path for the state dict 
                pickle.

            :image_logging: callable int -> None, handles logging an image
                visualization of performance to tensorboard if passed.

            :device: int, gpu number.

            :max_iterations: int, total number of minibatches sampled from the
                dataloader, irrespective of dataset size.

            :init_state_path: str or Path, path to state dict to be loaded 
                before training begins.
        """

        if init_state_path is not None:
            self.load_state(init_state_path)

        # Instantiate an Adam optimizer with no history.
        optimizer = self.create_new_optimizer()

        # Set mode to train, in case any layers require it.
        self.model.train()
    
        iteration = 0
        pbar = tqdm(desc="Training...", total=max_iterations)
        while True:
    
            for img, _ in dataloader:
                img = img.to(device)
    
                x_mu, x_chol, z_mu, z_logvar = self.model(img)
                
                # This is the different bit from the torch_supn trainer.
                if self.enforce_l2:
                    x_chol = torch.zeros_like(x_chol)
                    x_chol[:,0] = torch.ones(x_chol.shape[-1])

                    if self.stdev is not None:
                        x_chol = x_chol / self.stdev

                loss_val = self.elbo_loss(img, x_mu, x_chol, z_mu, z_logvar)
    
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
    
                if image_logging is not None: 
                    with torch.no_grad():
                        image_logging( iteration )
    
                if iteration > max_iterations: break
    
                iteration += 1
                pbar.update()
    
            if iteration > max_iterations: break

        self.model.eval()
        if save_state_path is not None: self.save_state(save_state_path)


class WaveletVAETrainer:
    r"""
    Class that handles the training of the Wavelet VAE. Uses the Adam optimizer
    and negative L2 loss by default. Supports some basic logging and tries to 
    expose very few parameters.
    """

    def __init__(self, model, 
            summary_writer=None,
            learning_rate=1e-3,
            stdev=None):
        r"""
        Args:

            :model: torch.Module, the Wavelet VAE.

            :summary_writer: torch.utils.tensorboard.SummaryWriter object, used
                to log loss values and image reconstructions optionally. It is
                passed to the loss function, which handles the io to the logger
                separately.

            :learning_rate: float

            :stdev: float or None, if enforce_l2 is True, then we can further
                weigh the KL and L2 terms. This parameter is the standard 
                deviation, and will enter as a 1/stdev multiplier for the fake
                Cholesky decomposition.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.stdev = stdev

        if summary_writer is not None:
            listener = LoggingScalarListener(summary_writer)
        else:
            listener = None

        if stdev is None:
            self.loss = L2VAELoss(loss_logging_listener = listener)
        else:
            self.loss = L2VAELoss(loss_logging_listener = listener, stdev=stdev)

    def create_new_optimizer(self):
        r"""

        Returns:
            torch.optim.Adam optimizer.
        """
        optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate)

        return optimizer

    def load_state(self, state_path):
        r"""
        Load the state dict into the model.

        Args:
            :state_path: Path
        """
        state_dict = torch.load(state_path)
        self.model.load_state_dict(state_dict)

    def save_state(self, state_path):
        r"""
        Save the current model's state dict.

        Args:
            :state_path: Path
        """
        torch.save(self.model.state_dict(), state_path)

    def train(self, dataloader, save_state_path, 
            image_logging=None, 
            device=0, max_iterations=10000, 
            init_state_path=None):
        r"""
        Runs the training loop and saves the final state dict.

        Args:

            :dataloader: torch.utils.data.DataLoader object which wraps the
                training dataset.

            :save_state_path: path or str, output path for the state dict 
                pickle.

            :image_logging: callable int -> None, handles logging an image
                visualization of performance to tensorboard if passed.

            :device: int, gpu number.

            :max_iterations: int, total number of minibatches sampled from the
                dataloader, irrespective of dataset size.

            :init_state_path: str or Path, path to state dict to be loaded 
                before training begins.
        """

        if init_state_path is not None:
            self.load_state(init_state_path)

        # Instantiate an Adam optimizer with no history.
        optimizer = self.create_new_optimizer()

        # Set mode to train, in case any layers require it.
        self.model.train()
    
        iteration = 0
        pbar = tqdm(desc="Training...", total=max_iterations)
        while True:
    
            for img, _ in dataloader:
                img = img.to(device)
    
                x_mu, z_mu, z_logvar = self.model(img)
                
                loss_val = self.loss(img, x_mu, z_mu, z_logvar)
    
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
    
                if image_logging is not None: 
                    with torch.no_grad():
                        image_logging( iteration )
    
                if iteration > max_iterations: break
    
                iteration += 1
                pbar.update()
    
            if iteration > max_iterations: break

        self.model.eval()
        if save_state_path is not None: self.save_state(save_state_path)
