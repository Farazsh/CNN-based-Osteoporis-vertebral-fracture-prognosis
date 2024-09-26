import multiprocessing
import click

import torch

from math import floor
from src.data_processing.convert_dicom_to_nii import start_conversion
from src.evaluate_checkpoint import start_checkpoint_evaluation
from src.hyperparameter_search import start_hp_search
from src.train import start_training
from src.lin_eval_and_fine_tune import start_lin_eval_fine_tune
from configs import config as c


@click.group()
def cli():
    pass


@cli.command()
@click.option("--num_gpus",
              required=False,
              default=torch.cuda.device_count(),
              help="The number of GPUs used for training one model.")
@click.option("--num_cpus",
              required=False,
              default=multiprocessing.cpu_count(),
              help="The number of CPU cores for training one model."
                   "Each gpu will have floor(num_cpus/num_gpus) number of cpus")
def train(num_gpus: int, num_cpus: int):
    """Initializes and starts the training."""
    num_cpus = floor(num_cpus/num_gpus)
    main_config = c.Config(num_cpus)
    for i in range(main_config.model.number_of_experiments):
        main_config.model.experiment_name = main_config.model.experiment_name[:-1]  + str(i)
        start_training(num_gpus, main_config)



@cli.command()
@click.option("--num_gpus",
              required=False,
              default=torch.cuda.device_count(),
              help="The number of GPUs used by the HP search.")
@click.option("--gpus_per_trial",
              required=False,
              default=1,
              help="The number of GPUs each trial will have. "
                   "Can be smaller than 1 to allow concurrent trials on a single gpu.")
@click.option("--num_cpus",
              required=False,
              default=multiprocessing.cpu_count(),
              help="The number of CPU cores used by the HP search.")
def hp_search(num_gpus: int, gpus_per_trial:float, num_cpus: int,):
    """Initializes and starts a hyperparameter search."""
    start_hp_search(num_gpus, gpus_per_trial, num_cpus)


@cli.command()
@click.option("--name",
              required=False,
              help="The name of the .ckpt file inside the checkpoints/ directory.")
@click.option("--dataset",
              required=False,
              default='db_ve',
              help="The dataset to evaluate the model on. Currently 'verse' and 'db' are supported")
@click.option("--validate",
              required=False,
              default=True,
              help="Whether to evaluate on the validation dataset.")
@click.option("--test",
              required=False,
              default=False,
              help="Whether to evaluate on the test dataset.")
def evaluate(name: str, dataset: str, validate: bool, test: bool):
    """Loads a checkpoint and evaluates it with testing."""
    start_checkpoint_evaluation(name, dataset, validate, test)


@cli.command()
@click.option("--name",
              required=False,
              help="The name of the experiment folder.")
@click.option("--linear_eval",
              required=False,
              default=False,
              help="Whether to do linear evaluation of the provided model")
@click.option("--fine_tune",
              required=False,
              default=True,
              help="Whether to fine tune the provided model")
def fine_tune(name: str, linear_eval: bool, fine_tune: bool):
    """Loads a checkpoint and evaluates it with testing."""
    start_lin_eval_fine_tune(name, linear_eval, fine_tune)



@cli.command()
@click.option("--root_dir",
              required=True,
              help="The root dir that contains further dirs with the dicom data")
@click.option("--output_dir",
              required=True,
              help="The output dir, where the nii.gz files will be saved")
@click.option("--progress",
              required=False,
              default=True,
              help="Show conversion progress")
def convert_dicom_to_nii(root_dir: str, output_dir: str, progress: bool):
    """Converts all dicom files inside the root directory to nii files """
    start_conversion(root_dir, output_dir, progress)



if __name__ == '__main__':
    # Entrypoint for the training
    torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    cli()
