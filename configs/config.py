# import torch
# import yaml
from pathlib import Path
# from src.io_tools import PROJECT_ROOT_DIR, TEST_RESULTS_DIR, CHECKPOINTS_DIR

# from src.nets.archs.resnet import ResNet50
# from src.nets.archs.senet import SerResNext50, SeResNet50, SerResNext101, SeResNet101, SeResNet152

WANDB_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
WANDB_PROJECT_NAME = "Vertebral_Fracture_Diagnostics_VerSe"

class DataModuleConfig:
    def __init__(self, num_cpus, simlr=False):
        self.img_dir = r"verse19/full_dataset_patches"
        self.labels_file = r"verse19/verse19_0vs23_splitted.csv"
        self.save_folder = "checkpoints/VerSe_exps"
        self.metric_filename = "20260329_VerSe_model_SEResNetBYOL100.xlsx"
        self.metrics_file = rf"{self.save_folder}/{self.metric_filename}"
        self.batch_size = 32
        self.accumulate_batches = 1
        self.num_workers = num_cpus
        self.label = "0vs23"
        self.censor_label = 'Censored_label'
        self.ID_label = 'ID'
        self.input_size = 605040
        self.input_dimension = 3
        self.split_name = "Dataset"
        self.Inference_only = False

        self.image_augmentation_prob = 0.5
        self.augmentation_magnitude = 8  # must at least be one
        self.num_sequential_transforms = 3


class HPConfig:
    def __init__(self):
        self.hp_run = False  # set this to true when doing a hp search
        self.hp_class = 0
        self.name = 'SelfDistillRes50GS_real'#'UnsupFinetuneSENetLRSearchFracture'
        self.metrics_to_optimize = 'f1'  # multiclass 'f1'
        self.mode = 'max'
        self.num_samples = 131072
        self.reduction_factor = 4
        self.max_time_units_per_trial = 32
        self.grace_period = 1
        self.points_to_evaluate = []
        self.local_dir = Path('/opt/checkpoints')
        self.resume = False
