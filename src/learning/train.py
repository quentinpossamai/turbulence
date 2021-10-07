import argparse
import datetime
# import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
import torch
from torch.utils.data import DataLoader

from src.data.euroc_mav.loading import EurocMAVDatasetHorizon
from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
# from src.drone_models.residual.groundthruth import GroundTruth  # Sanity check
# from src.drone_models.residual.mlp import MLPResidual
# from src.drone_models.residual.cnn_lightning import CNNResidual
# from src.drone_models.residual.gru_cnn import GRUCNNResidual
from src.drone_models.residual.presence_classifier import PresenceClassifier
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

EUROC_MAV_RESIDUAL = GENERAL_ANALYSIS_PATH + "euroc_mav_residual_training/"
mkdir_if_not_exist([EUROC_MAV_RESIDUAL])

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
GPU_COUNT = int(device != torch.device("cpu"))

print(f"{device=}")
print(f"{GPU_COUNT=}")


def main():
    # Args are hyper parameters to optimize
    parser = argparse.ArgumentParser(description="Identify model on data.")
    # Program args
    parser.add_argument('--batch_size', default=64, type=int, help="Size of a batch")
    parser.add_argument('--load_weights', default=None, type=str, help="Path to previous weights")
    class_name = PresenceClassifier
    # Module args
    parser = class_name.add_model_specific_args(parser)
    # Trainer args
    parser = pl.Trainer.add_argparse_args(parser)  # accelerator, max_epochs
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    version = (f"{now}_"
               f"{class_name.__name__}_"
               f"{args.activation}_"
               f"bs{args.batch_size}_"
               f"{args.embedding}"
               # f"h{args.horizon}_"
               f"l{args.layer}_"
               f"lr{args.lr}_"
               f"epch{args.max_epochs}_"
               f"p{args.past}_"
               f"us{args.unit_size}_"
               )
    training_path = EUROC_MAV_RESIDUAL + f"{class_name.__name__}_{args.embedding}/"

    print(f"{training_path=}")
    print(f"{version=}")

    # Reproducibility
    seed = 0
    seed_everything(seed=seed, workers=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=training_path, version=version, log_graph=True)
    # analytic_model = DynamicalModel(param="identification", dtype=torch.float32)
    if args.load_weights is None:
        run = class_name(
            activation=args.activation,
            embedding=args.embedding,
            layer=args.layer,
            learning_rate=args.lr,
            # model=analytic_model,
            motor_size=6,
            past=args.past,
            unit_size=args.unit_size,
        )
    else:
        print(f"\nReloading weights from: {args.load_weights}")
        run = class_name.load_from_checkpoint(
            args.load_weights,
            activation=args.activation,
            embedding=args.embedding,
            layer=args.layer,
            learning_rate=args.lr,
            # model=analytic_model,
            motor_size=6,
            past=args.past,
            unit_size=args.unit_size,
        )

    # num_workers = 0
    num_workers = 12 if GPU_COUNT else 4
    # Datasets
    train_data = DataLoader(EurocMAVDatasetHorizon("train",
                                                   traj_len=args.past + args.horizon,
                                                   dtype=torch.float32,
                                                   batch_dim=False),
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True
                            )

    validation_data = DataLoader(EurocMAVDatasetHorizon("validation",
                                                        traj_len=args.past + args.horizon,
                                                        dtype=torch.float32,
                                                        batch_dim=False),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True
                                 )

    trainer = pl.Trainer(gpus=GPU_COUNT,
                         logger=tb_logger,
                         deterministic=True,
                         max_epochs=args.max_epochs,
                         # fast_dev_run=False,
                         # track_grad_norm=2,
                         # accelerator=None,
                         accelerator='ddp' if GPU_COUNT else None,
                         log_gpu_memory=True,
                         )
    trainer.fit(run, train_data, validation_data)


if __name__ == "__main__":
    main()
