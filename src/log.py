from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, writer_name, saving_path):
        # Loss save
        self.valid_loss_log = []  # All loss
        self.valid_batch_loss_log = []

        self.train_loss_log = []  # All loss
        self.train_batch_loss_log = []

        # Indexes
        self.g_idx = 0  # Global index (Number of back propagation).
        self.epoch = 0  # Current epoch index
        self.tb_idx = None  # Training current batch index
        self.vb_idx = None  # Validation current batch index

        # Tensorboard
        print("Tensorboard logdir: ", writer_name)
        self.sw = SummaryWriter(writer_name)  # tensorboard writer

        # Saving path
        self.saving_path = None
