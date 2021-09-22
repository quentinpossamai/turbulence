from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_name, saving_path):
        self.run_name = run_name

        # Loss save
        self.valid_loss_log = []  # All loss
        self.valid_epoch_loss_log = []

        self.train_loss_log = []  # All loss
        self.train_epoch_loss_log = []

        # Indexes
        self.g_idx = 0  # Global index (Number of back propagation).
        self.epoch = 0  # Current epoch index
        self.tb_idx = None  # Training current batch index
        self.vb_idx = None  # Validation current batch index

        # Tensorboard
        print("Tensorboard logdir: ", saving_path)
        self.sw = SummaryWriter(saving_path)  # tensorboard writer

        # Saving path
        self.saving_path = saving_path
