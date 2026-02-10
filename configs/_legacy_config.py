import torch
import yaml
from pathlib import Path
from src.io_tools import PROJECT_ROOT_DIR, TEST_RESULTS_DIR, CHECKPOINTS_DIR

from src.nets.archs.resnet import ResNet50
from src.nets.archs.senet import SerResNext50, SeResNet50, SerResNext101, SeResNet101, SeResNet152


class DataModuleConfig:
    def __init__(self, num_cpus):
        self.img_dir = r"MrOs_dataset/patches_arbitary_sized"  # MrOS

        # self.labels_file = "data/db/labels_new_splitted.csv"  # DiagBilanz
        self.labels_file = r"MrOs_dataset/MrOs_Label_from_meta_v12_1_relabelled2_newsplit.csv"  # MrOS
        self.batch_size = 16
        self.accumulate_batches = 1
        self.num_workers = num_cpus
        # possible labels are Filename;Score;DDC;Fractured;0vs123;01vs23;01vs2vs3
        self.label = "IF10SQ1"
        self.censor_label = 'Censored_label'
        self.ID_label = 'ID'
        self.input_size = 47
        self.input_dimension = 3
        self.split_name = "Split1"
        self.Inference_only = False
        self.test_as_validation = False  #todo: Be careful of this setting || only set True from main script

        # 0.33 always
        self.image_augmentation_prob = 0.33

        # 5 for supervvised fracture
        # 7 for supervised gs
        # 5 for unsup finetune gs
        # 5 for unsupervised fracture
        # 2 for unsup SE GS
        # 8 for unsup finetune SE fracture
        self.augmentation_magnitude = 8  # must at least be one

        # 3 for supervised fracture
        # 2 for supervised gs
        # 4 for unsup finetune gs
        # 7 for unsupervised frarcture
        # 8 for unsup SE GS
        # 6 for unsup se finetune fracture
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
