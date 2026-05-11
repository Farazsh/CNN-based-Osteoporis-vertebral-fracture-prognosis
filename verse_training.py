import monai.networks.nets
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from configs.config import DataModuleConfig, WANDB_KEY, WANDB_PROJECT_NAME
from src.datasets.dataloader import DataModule
import warnings
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torchmetrics.classification import BinaryAUROC, BinarySpecificity, BinaryF1Score, BinaryConfusionMatrix, \
    BinaryAccuracy, BinaryPrecision, BinaryAveragePrecision
import torch
import torch.optim as optim
import wandb
import os
import random
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torchmetrics import Metric
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision.models.feature_extraction import create_feature_extractor
# from torchvision.models import resnet50
# from MedicalNetResnet import resnet50
import torch.nn.functional as F
import re
import monai
from os.path import join
import glob
import json
from copy import deepcopy
from auxillary_functions import plot_auroc, count_and_print_num_of_parameters,\
    plot_sigmoidVsFractureTime, box_strip_plot, plot_auroc_auprc
from src.nets.archs.fnet_2d_3d import Fnet2D, Fnet3D
from src.nets.archs.resnet50_3d import resnet50_3d
from src.nets.archs.resnet50_2d import resnet50_2d
from src.nets.archs.resnet2d import Resnet18_2d
from src.nets.archs.resnet3d import Resnet18_3d
from src.nets.archs.resnet_family import generate_model

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['figure.dpi'] = 200
optimizers = {"adam": Adam, "sgd": SGD, "adamw": AdamW}

from torchvision.models import resnet50 as torch_resnet_50, ResNet50_Weights
from torchvision.models import resnet18 as torch_resnet_18, ResNet18_Weights
from src.nets.archs.senet import CustomSEResNet50
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class BaseModel(pl.LightningModule):
    def __init__(self, class_ratio, spatial_dims, learning_rate, network, optimizer_name, threshold, threshold_tuning,
                 save_folder, inference_only, df, split, fold, label_name, fracture_timeline, label_suffix,
                 evaluate_ckpt, metrics_file, epoch):
        super().__init__()


        # Tuning parameters
        self.network = network
        self.spatial_dims = spatial_dims
        self.lr = learning_rate
        self.optimizer = optimizers[optimizer_name]
        self.class_ratio = class_ratio
        self.threshold = threshold
        self.threshold_tuning = threshold_tuning
        self.swa_epoch_start = 3
        self.perform_swa = False
        self.num_averaged_models = 0

        # Default parameters
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([self.class_ratio]))
        self.sigmoid_layer = torch.nn.Sigmoid()
        self.alpha = 0.2

        # For monitoring performance
        self.train_metric_dict = {}
        self.validation_metric_dict = {}
        self.test_metric_dict = {}
        self.model_outputs = ['predictions', 'actuals']
        self.tuned_threshold = []
        self.epoch_no = epoch
        self.total_steps = 0
        self.save_folder = save_folder
        self.metrics_file = metrics_file

        # Inference only
        self.inference_only = inference_only
        self.label_name = label_name
        self.fracture_timeline = fracture_timeline
        self.suffix = label_suffix
        self.evaluate_ckpt = evaluate_ckpt

        # For recording model outputs for each datapoint
        if inference_only and df is not None:
            self.model_outputs = ['predictions', 'actuals', 'logits', 'id']
            self.df = df
            self.split = split
            self.fold = fold
            # self.df = self.df[self.df[f"Split{self.split}"]=='Testing']  # MrOS
            self.df = self.df[self.df[f"Dataset"] == 'Testing']
            self.df['ID'] = self.df['ID'].astype(int)

        for op in self.model_outputs:
            self.train_metric_dict[op] = torch.empty(1, 0)
            self.validation_metric_dict[op] = torch.empty(1, 0)
            self.test_metric_dict[op] = torch.empty(1, 0)

        if spatial_dims == 3:
            if self.network.lower() == 'fnet':
                self.backbone = Fnet3D()
            elif self.network.lower() == 'seresnext50':
                self.backbone = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=1, num_classes=1)
            elif self.network.lower() == "resnet18new":
                self.backbone = generate_model(model_depth=18, n_input_channels=1, n_classes=1)
            elif self.network.lower() == "resnet18newpt":
                self.backbone = generate_model(model_depth=18, n_input_channels=1, n_classes=1)
                pretrain_path = r"medical_net/MedicalNet_pytorch_files2/pretrain/resnet_18_23dataset.pth"
                print('loading pretrained model {}'.format(pretrain_path))
                pretrain = torch.load(pretrain_path)
                net_dict = self.backbone.state_dict()
                pretrain_dict = {}
                for k, v in pretrain['state_dict'].items():
                    if k[7:] in net_dict:
                        pretrain_dict[k[7:]] = v
                    else:
                        print(f"Not found {k}")
                net_dict.update(pretrain_dict)
                self.backbone.load_state_dict(net_dict, strict=True)
                print("Pretrained weigths succesfully loaded")
            elif self.network.lower() == "resnet50new":
                self.backbone = generate_model(model_depth=50, n_input_channels=1, n_classes=1)
            elif self.network.lower() == "resnet50newpt":
                self.backbone = generate_model(model_depth=50, n_input_channels=1, n_classes=1)
                pretrain_path = r"medical_net/MedicalNet_pytorch_files2/pretrain/resnet_50_23dataset.pth"
                print('loading pretrained model {}'.format(pretrain_path))
                pretrain = torch.load(pretrain_path)
                net_dict = self.backbone.state_dict()
                pretrain_dict = {}
                for k, v in pretrain['state_dict'].items():
                    if k[7:] in net_dict:
                        pretrain_dict[k[7:]] = v
                    else:
                        print(f"Not found {k}")
                net_dict.update(pretrain_dict)
                self.backbone.load_state_dict(net_dict, strict=True)
                print("Pretrained weigths succesfully loaded")
            elif self.network.lower() == 'resnet18':
                self.backbone = Resnet18_3d()
            elif self.network.lower() == 'resnet50':
                self.backbone = resnet50_3d()
            elif self.network.lower() == 'densenet201':
                self.backbone = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=1, out_channels=1)
            elif self.network.lower() == 'seresnetsll':
                self.backbone = monai.networks.nets.SEResNet50(input_3x3=False, in_channels=1, spatial_dims=3,
                                                               dropout_prob=0.5, num_classes=1)
                pretrained_state_dict = torch.load(r"jl_ckpts/0_rtn_ckpt_epoch=255_step=256.00.ckpt")['state_dict']
                pretrained_state_dict_corrected = {key.split('.', 1)[1]: value for key, value in
                                                   pretrained_state_dict.items()}
                self.backbone.load_state_dict(pretrained_state_dict_corrected, strict=False)
            elif self.network.lower() == 'seresnetbyol100':
                self.backbone = CustomSEResNet50()
                self._load_backbone_from_ckpt(
                    ckpt_path=r"ca_ckpts/0_rtn_ckpt_epoch=99_step=98.00_v2.ckpt",
                    strict=False,
                )
            elif self.network.lower() == 'seresnet50':
                self.backbone = monai.networks.nets.SEResNet50(input_3x3=False, in_channels=1, spatial_dims=3,
                                                               dropout_prob=0.5, num_classes=1)
            elif self.network.lower() == "resnet10new":
                self.backbone = generate_model(model_depth=10, n_input_channels=1, n_classes=1)
            elif self.network.lower() == "resnet18new":
                self.backbone = generate_model(model_depth=18, n_input_channels=1, n_classes=1)
            elif self.network.lower() == "resnet50new":
                self.backbone = generate_model(model_depth=50, n_input_channels=1, n_classes=1)
            else:
                print('Not Implemented', self.network)

        if spatial_dims == 2:
            print('Not Implemented', self.network)

        if self.perform_swa:
            self.avg_model = deepcopy(self.backbone)

    def _load_backbone_from_ckpt(
            self,
            ckpt_path: str,
            prefixes=("backbone.", "model.backbone.", "encoder.", "model.encoder."),
            strict: bool = False,
    ):
        """
        Loads ONLY backbone weights from a Lightning checkpoint or a raw state_dict.

        - Supports Lightning .ckpt with {"state_dict": ...}
        - Filters keys by prefixes, then strips prefix so it matches self.backbone.*
        """
        if ckpt_path is None or str(ckpt_path).strip() == "":
            print("[pretrained] No checkpoint path provided -> random init backbone.")
            return

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        # 1) if it's already a raw backbone state_dict (no prefixes), try direct load
        #    (this will succeed if keys look like 'layer0.conv1.weight', etc.)
        direct_ok = any(k.startswith("layer") or k.startswith("features") for k in state.keys())
        if direct_ok:
            missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
            print(
                f"[pretrained] Loaded backbone directly from state_dict. missing={len(missing)} unexpected={len(unexpected)}")
            return

        # 2) otherwise filter by prefix
        selected = {}
        used_prefix = None
        for pfx in prefixes:
            tmp = {k[len(pfx):]: v for k, v in state.items() if k.startswith(pfx)}
            if len(tmp) > 0:
                selected = tmp
                used_prefix = pfx
                break

        if len(selected) == 0:
            # helpful debug: show a few keys
            example_keys = list(state.keys())[:20]
            raise RuntimeError(
                "[pretrained] Could not find backbone weights in checkpoint.\n"
                f"Tried prefixes: {prefixes}\n"
                f"Example checkpoint keys: {example_keys}"
            )

        missing, unexpected = self.backbone.load_state_dict(selected, strict=strict)
        print(
            f"[pretrained] Loaded backbone from '{ckpt_path}' using prefix '{used_prefix}'. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    def change_model_state_to_inference(self, df, split, fold, label_name, fracture_timeline):
        self.inference_only = True
        self.label_name = label_name
        self.fracture_timeline = fracture_timeline
        self.split = split
        self.fold = fold
        self.df = df
        self.df = self.df[self.df[f"Split{self.split}"] == 'Testing']
        self.df['ID'] = self.df['ID'].astype(int)

        self.model_outputs = ['predictions', 'actuals', 'id']
        for op in self.model_outputs:
            self.train_metric_dict[op] = torch.empty(1, 0)
            self.validation_metric_dict[op] = torch.empty(1, 0)
            self.test_metric_dict[op] = torch.empty(1, 0)

    def mixup_data(self, x, y):
        """ Returns mixed inputs, pairs of targets, and lambda
        reference: mixup: Beyond Empirical Risk Minimization
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_loss(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a.float()) + (1 - lam) * self.loss_fn(pred, y_b.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.classifier(self.backbone(x))
        return self.backbone(x)

    def configure_optimizers(self):
        # Fixed LR
        return self.optimizer(self.parameters(), lr=self.lr)
        # return self.optimizer(self.parameters(), lr=self.lr, weight_decay=1e-3)

        # Cosine Annealing WarmRestarts Scheduler
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, eta_min=0)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

        # Cosine Annealing Scheduler
        # optimizer = self.optimizer(self.parameters(), lr=self.lr)
        # scheduler = CosineAnnealingLR(optimizer, T_max=32, eta_min=5e-4)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def _step(self, batch):
        x, y = batch
        preds_linear_layer = self(x).flatten()  # For stability in loss calculation, output is taken from linear layer
        loss = self.loss_fn(preds_linear_layer, y.float())
        preds = self.sigmoid_layer(preds_linear_layer)  # For comparison to real labels, output is through sigmoid layer
        return loss, {"predictions": preds.resize(1, len(preds)).to('cpu'), "actuals": y.resize(1, len(y)).to('cpu')}

        # Mix-up
        # x, y = batch
        # mixed_x, y_a, y_b, lam = self.mixup_data(x, y)  # apply mixup
        # preds = self(mixed_x).flatten()  # pass images to model
        # loss = self.mixup_loss(preds, y_a, y_b, lam)  # calculate loss
        # return loss, {"predictions": preds.resize(1, len(preds)).to('cpu'), "actuals": y.resize(1, len(y)).to('cpu')}

    def _swa_step(self, batch):
        """
        Takes output from Averaged model after the swa starts
        """
        x, y = batch

        if self.epoch_no > self.swa_epoch_start:
            preds_linear_layer = self.avg_model(x).flatten()
        else:
            preds_linear_layer = self(x).flatten()
        loss = self.loss_fn(preds_linear_layer, y.float())

        preds = self.sigmoid_layer(preds_linear_layer)  # For comparison to GT actual output is through sigmoid layer
        return loss, {"predictions": preds.resize(1, len(preds)).to('cpu'), "actuals": y.resize(1, len(y)).to('cpu')}

    def _inference_step(self, batch):
        """
        Notes the ids along with actuals and predictions in case of inference
        """
        x, y = batch
        preds_linear_layer = self(x).flatten()
        y_labels = y[0].float()
        y_ids = y[1]
        loss = self.loss_fn(preds_linear_layer, y_labels)
        preds = self.sigmoid_layer(preds_linear_layer)
        return loss, {"predictions": preds.resize(1, len(preds)).to('cpu'),
                      "actuals": y_labels.resize(1, len(y_labels)).to('cpu'),
                      "logits": preds_linear_layer.resize(1, len(preds_linear_layer)).to('cpu'),
                      "id": y_ids.resize(1, len(y_ids)).to('cpu')}

    def training_step(self, batch, batch_idx):
        loss, calculated_metrics = self._step(batch)
        self.log("train_loss", loss)

        for extra_met in self.model_outputs:
            self.train_metric_dict[extra_met] = torch.concat(
                (self.train_metric_dict[extra_met], calculated_metrics[extra_met]), axis=1)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.perform_swa:
            loss, calculated_metrics = self._swa_step(batch)
        else:
            loss, calculated_metrics = self._step(batch)

        self.log("validation_loss", loss)
        for extra_met in self.model_outputs:
            self.validation_metric_dict[extra_met] = torch.concat(
                (self.validation_metric_dict[extra_met], calculated_metrics[extra_met]), axis=1)

        return loss

    def test_step(self, batch, batch_idx):
        if self.inference_only:
            loss, calculated_metrics = self._inference_step(batch)
        else:
            loss, calculated_metrics = self._step(batch)

        self.log("test_loss", loss)
        for extra_met in self.model_outputs:
            self.test_metric_dict[extra_met] = torch.concat(
                (self.test_metric_dict[extra_met], calculated_metrics[extra_met]), axis=1)

        return loss

    def compute_metrics(self, prediction, target, dataset):
        target = target.long()
        bin_auroc = BinaryAUROC()
        auroc = bin_auroc(prediction.flatten(), target.flatten())

        bin_auprc = BinaryAveragePrecision()
        auprc = bin_auprc(prediction.flatten(), target.flatten())
        preds = (prediction >= self.threshold).long()
        tp = torch.sum(target * preds).item()
        tn = torch.sum((1 - target) * (1 - preds)).item()
        fp = torch.sum((1 - target) * preds).item()
        fn = torch.sum(target * (1 - preds)).item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-15)
        precision = tp / (tp + fp + 1e-15)
        specificity = tn / (tn + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-15)

        self.log(f"{dataset}_accuracy", accuracy)
        self.log(f"{dataset}_precision", precision)
        self.log(f"{dataset}_specificity", specificity)
        self.log(f"{dataset}_recall", recall)
        self.log(f"{dataset}_f1", f1)
        self.log(f"{dataset}_auroc", auroc)
        self.log(f"{dataset}_auprc", auprc)

        gt_values = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        for gt in gt_values:
            self.log(f"{dataset}_{gt}", gt_values[gt])

        if dataset == 'test':
            # Record metric scores in an external excel file
            # f"20260329_{net}_3D_PrevalentFractures_{label_name}_605040_AdamWFixedLR{lr}_outerloop_{seed}"
            df = pd.read_excel(self.metrics_file, index_col=None)
            data = self.save_folder.split('/')[2].split('_')
            meta_data_length = len(df.columns) - 9  # 7 is number of metrics, 2 for split and epochs
            if len(data) > meta_data_length:
                print('Inconsistent model name, cropping the ends')
                data = data[:meta_data_length]
            elif len(data) < meta_data_length:
                print('Missing variables in model name, appending blanks')
                data.extend(['']*(meta_data_length-len(data)))
            data.extend([self.save_folder.split('/')[3][-1], self.epoch_no])
            data.extend([auroc.item(), auprc.item(), accuracy, precision, recall, f1, specificity])
            df.loc[len(df)] = data
            df.to_excel(self.metrics_file, index=False)

    def compute_metrics_epoch(self, prediction, target, dataset):
        target = target.long()
        bin_auroc = BinaryAUROC()
        auroc = bin_auroc(prediction.flatten(), target.flatten())

        bin_auprc = BinaryAveragePrecision()
        auprc = bin_auprc(prediction.flatten(), target.flatten())
        preds = (prediction >= self.threshold).long()
        tp = torch.sum(target * preds).item()
        tn = torch.sum((1 - target) * (1 - preds)).item()
        fp = torch.sum((1 - target) * preds).item()
        fn = torch.sum(target * (1 - preds)).item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-15)
        precision = tp / (tp + fp + 1e-15)
        specificity = tn / (tn + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-15)

        self.log(f"{dataset}_accuracy_epoch", accuracy)
        self.log(f"{dataset}_precision_epoch", precision)
        self.log(f"{dataset}_specificity_epoch", specificity)
        self.log(f"{dataset}_recall_epoch", recall)
        self.log(f"{dataset}_f1_epoch", f1)
        self.log(f"{dataset}_auroc_epoch", auroc)
        self.log(f"{dataset}_auprc_epoch", auprc)
        self.log(f"{dataset}_threshold_epoch", self.threshold)

        gt_values = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        for gt in gt_values:
            self.log(f"{dataset}_{gt}_epoch", gt_values[gt])

    def find_threshold_for_recall(self, y_true, y_pred) -> None:
        """
        Find a threshold that achieves a desired value of recall example 90
        """

        target_recall = 0.90
        # Combine y_true and y_pred into a single array and sort by y_pred descending
        combined = sorted(zip(y_pred, y_true), reverse=True)
        # Initialize counters
        tp = 0  # True Positives
        fn = sum(y_true)  # Initially, we assume all positives are not detected

        for threshold, (pred, actual) in enumerate(combined):
            if actual == 1:
                tp += 1
                fn -= 1
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if recall >= target_recall:
                # Return the threshold value that achieves the target recall
                # The threshold is set at the y_pred value at the current position
                # return combined[threshold][0]  # Return the y_pred value as threshold
                self.threshold = combined[threshold][0]
                return

        # If we exit the loop without achieving the target recall, return the last threshold tried
        self.threshold = combined[-1][0] if combined else 0
        return
        # return combined[-1][0] if combined else 0

    def on_train_epoch_end(self) -> None:

        prediction, target = self.train_metric_dict['predictions'].detach(), self.train_metric_dict['actuals'].detach()
        # self.find_threshold_for_recall(target.flatten().numpy(), prediction.flatten().numpy())
        # _ = plot_auroc(target.flatten().numpy(), prediction.flatten().numpy(), f'Training_Epoch_{self.epoch_no}', self.threshold, self.save_folder)
        self.compute_metrics_epoch(prediction, target, "train")
        self.train_metric_dict['predictions'] = torch.empty(1, 0)
        self.train_metric_dict['actuals'] = torch.empty(1, 0)
        self.epoch_no += 1

        if self.perform_swa and (self.epoch_no > self.swa_epoch_start):
            # Computes average of swa model and current model and updated params of both with the new avg
            for p_swa, p_model in zip(self.avg_model.parameters(), self.backbone.parameters()):
                updated_param = p_swa + (p_model - p_swa) / (self.num_averaged_models + 1)
                p_swa.detach().copy_(updated_param)
                # p_model.detach().copy_(updated_param)
            self.num_averaged_models += 1

    def on_train_end(self) -> None:
        if self.perform_swa:
            for src_param, dst_param in zip(self.avg_model.parameters(), self.backbone.parameters()):
                dst_param.detach().copy_(src_param.to(dst_param.device))

    def on_validation_epoch_end(self) -> None:

        prediction, target = self.validation_metric_dict['predictions'].detach(), self.validation_metric_dict[
            'actuals'].detach()
        if self.threshold_tuning:
            val_threshold = plot_auroc(target.flatten().numpy(), prediction.flatten().numpy(),
                                        f'Validation_Epoch_{self.epoch_no}', False, self.save_folder)
        else:
            val_threshold = self.threshold
        self.compute_metrics(prediction, target, "validation")
        self.validation_metric_dict['predictions'] = torch.empty(1, 0)
        self.validation_metric_dict['actuals'] = torch.empty(1, 0)
        self.tuned_threshold.append(round(val_threshold, 2))

        json_outfile = {'threshold': self.tuned_threshold, 'class_ratio': self.class_ratio}
        with open(join(self.save_folder, 'threshold_class_ratio.json'), "w") as outfile:
            json.dump(json_outfile, outfile)

    def on_test_epoch_end(self) -> None:

        if self.inference_only:
            prediction, target, logits, id = self.test_metric_dict['predictions'].detach(), \
                                             self.test_metric_dict['actuals'].detach(), \
                                             self.test_metric_dict['logits'].detach(), \
                                             self.test_metric_dict['id'].detach()
            self.compute_inference(prediction, target, logits, id)
            return

        prediction, target = self.test_metric_dict['predictions'].detach(), self.test_metric_dict['actuals'].detach()
        # _ = plot_auroc(target.flatten().numpy(), prediction.flatten().numpy(), f'Testing_Epoch_{self.epoch_no}',
        #                self.threshold, self.save_folder)
        # plot_auroc_auprc(target.flatten().numpy(), prediction.flatten().numpy(), self.save_folder, self.split, '_Test')
        self.compute_metrics(prediction, target, "test")
        self.test_metric_dict['predictions'] = torch.empty(1, 0)
        self.test_metric_dict['actuals'] = torch.empty(1, 0)

    def compute_inference(self, prediction, target, logits, id) -> None:
        pred_binary = (prediction >= self.threshold).long().flatten().numpy()
        prediction = prediction.flatten().numpy()
        logits = logits.flatten().numpy()
        target = target.flatten().numpy()
        id = id.flatten().numpy().astype(int)
        mapper_sigmoid = dict(zip(id, prediction))
        mapper_logits = dict(zip(id, logits))

        if self.evaluate_ckpt:
            self.df[f"Sigmoid_Epoch{self.epoch_no}"] = self.df.ID.apply(lambda x: mapper_sigmoid[x])
            self.df.to_csv(join(self.save_folder, f"Checkpoint results {self.suffix}.csv"),
                           index=False, sep=';')
            return

        plot_auroc_auprc(target, prediction, self.save_folder, self.split, self.suffix)
        mapper_binary = dict(zip(id, pred_binary))
        # self.df[f"Sigmoid_split{self.split}"] = self.df.ID.apply(lambda x: mapper_sigmoid[x])  # MrOS
        # self.df[f"Binary_split{self.split}"] = self.df.ID.apply(lambda x: mapper_binary[x])  # MrOS
        self.df[f"Sigmoid"] = self.df.ID.apply(lambda x: mapper_sigmoid[x])  # VerSe
        self.df[f"Binary"] = self.df.ID.apply(lambda x: mapper_binary[x])  # VerSe
        self.df[f"Logits"] = self.df.ID.apply(lambda x: mapper_logits[x])  # VerSe
        self.df.to_csv(join(self.save_folder, f"{self.network}_{self.spatial_dims}D_sigmoid_results_fold_{self.fold}{self.suffix}.csv"),
                       index=False, sep=';')
        # box_strip_plot(self.df, self.split, self.label_name, self.save_folder)
        # plot_sigmoidVsFractureTime(self.df, self.split, self.save_folder, self.threshold, self.fracture_timeline, self.suffix)  # MrOS


class PLmodel(BaseModel):
    def __init__(self, class_ratio: float, spatial_dims: int, learning_rate: float, network: str = 'fnet',
                 optimizer_name: str = "adamw", threshold: float = 0.5, threshold_tuning = False,
                 save_folder: str = r"checkpoints/MrOS/dump", inference_only=False, df=None, split=0, fold=0,
                 label_name='XMSQGE2', fracture_timeline=5478, label_suffix='', evaluate_ckpt=False,
                 metrics_file=r"data/checkpoints/26112024_MrOS_model_results.xlsx", epoch=0):

        super(PLmodel, self).__init__(class_ratio, spatial_dims, learning_rate, network, optimizer_name,
                                      threshold, threshold_tuning, save_folder, inference_only, df, split, fold,
                                      label_name, fracture_timeline, label_suffix, evaluate_ckpt, metrics_file, epoch)

    def __str__(self):
        return 'PLmodel'


def start_training(model_name, network, max_epoch, lr, pretrained):

    net_architecture = network
    learning_rate = lr
    epochs = max_epoch
    spatial_dims = 3
    binary_label = '0vs23'
    threshold_tuning = False
    optimizer_name = 'adamw'
    monitoring_metric = 'validation_loss'
    monitoring_mode = 'min'

    project_name = WANDB_PROJECT_NAME
    group_name = model_name
    wandb.login(key=WANDB_KEY)
    wandb_logging_config = {
        'net_architecture': net_architecture,
        'learning_rate': learning_rate,
        'optimizer': optimizer_name,
        'epochs': epochs,
        'task_name': project_name,
        'monitoring_metric': monitoring_metric,
        'monitoring_mode': monitoring_mode,
    }

    config_file = DataModuleConfig(num_cpus=2, simlr=False)
    dataset_loader = DataModule(config_file)
    dataset_loader.prepare_data()
    class_ratio = dataset_loader.get_class_ratio()

    checkpoint_folder_main = rf"{config_file.save_folder}/{model_name}"
    os.makedirs(checkpoint_folder_main, exist_ok=True)
    with open(join(checkpoint_folder_main, 'training_info.json'), "w") as outfile:
        json.dump(wandb_logging_config, outfile)
    
    checkpoint_folder = rf"{config_file.save_folder}/{model_name}/Label{binary_label}"
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_filename = group_name + '_' + binary_label + '_top_ckpt_at_{epoch}_with_{validation_auroc:.3f}'
    experiment_name = binary_label
    
    wandb.init(project=project_name, name=experiment_name, config=wandb_logging_config, group=group_name)
    
    best_checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_folder,
                                               filename=checkpoint_filename,
                                               save_top_k=1,
                                               monitor=monitoring_metric,
                                               mode=monitoring_mode)

    model = PLmodel(class_ratio=class_ratio, spatial_dims=spatial_dims, learning_rate=learning_rate,
                    network=net_architecture, optimizer_name=optimizer_name, fold=binary_label,
                    threshold_tuning=threshold_tuning, save_folder=checkpoint_folder,
                    metrics_file=config_file.metrics_file)
    wandb_logger = WandbLogger(project=project_name, name=experiment_name)
    trainer = pl.Trainer(devices=1, max_epochs=epochs, logger=wandb_logger,
                         callbacks=[
                             best_checkpoint_callback,
                             EarlyStopping(monitor=monitoring_metric, min_delta = 0.005, patience=20, verbose=False, mode=monitoring_mode)
                             ]
                         )
    # trainer.fit(model=model, datamodule=dataset_loader)

    # For explicitly saving the checkpoint at the end of full-training
    # trainer.save_checkpoint(join(checkpoint_folder,group_name+'_'+split_name+f'Model_trained_for_epochs_{epochs}.ckpt'))

    best_model_path = glob.glob(join(checkpoint_folder, '*.ckpt'))[0]
    match = re.search(r"epoch=(\d+)", best_model_path)
    if match:
        epoch_number = match.group(1)
    else:
        epoch_number = 0

    model = PLmodel.load_from_checkpoint(best_model_path, class_ratio=class_ratio, spatial_dims=spatial_dims,
                                         network=net_architecture, learning_rate=learning_rate, threshold=0.5, label_name=binary_label,
                                         fold=binary_label, save_folder=checkpoint_folder, split=binary_label, epoch=epoch_number,
                                         metrics_file=config_file.metrics_file)
    trainer.test(model=model, datamodule=dataset_loader)
    wandb.finish()


def test_network():
    image_batch = torch.tensor(np.random.rand(32, 1, 60, 50, 40), dtype=torch.float32)
    label = torch.randint(0, 1, (32,))

    backbone = CustomSEResNet50()
    out = backbone(image_batch)
    count_and_print_num_of_parameters(backbone)
    print("Output Shape: ", out.shape)
    print("Labels Shape: ", label.shape)


def preprocess_and_inference(model_name, csv_filename, image_dir, fold_no, spatial_dims, net,
                             label_name, fracture_time, train_as_test=False, validation_as_test=False):
    split_no=4
    df = pd.read_csv(csv_filename, delimiter=';', index_col=None)
    split_name = "Dataset"   # 'Split4' for MrOS, 'Dataset' for VerSe
    config_file = DataModuleConfig(num_cpus=2, simlr=False)
    config_file.labels_file = csv_filename
    config_file.img_dir = image_dir
    config_file.split_name = "Dataset"   # 'Split4' for MrOS, 'Dataset' for VerSe
    config_file.label = label_name
    config_file.Inference_only = True
    config_file.input_dimension = spatial_dims
    dataset_loader = DataModule(config_file)
    dataset_loader.prepare_data()
    class_weight = dataset_loader.get_class_ratio()


    if train_as_test and validation_as_test:
        print("Cannot perform inference on train and validation set at once")
        return
    if train_as_test:
        dataset_loader.train_as_test = True
        df[split_name] = df[split_name].replace({'Training': 'Testing', 'Testing': 'Training'})
        save_label_suffix = '_Train'
    elif validation_as_test:
        dataset_loader.validation_as_test = True
        df[split_name] = df[split_name].replace({'Validation': 'Testing', 'Testing': 'Validation'})
        save_label_suffix = '_Validation'
    else:
        save_label_suffix = '_Test'
    save_folder = join("data/checkpoints/VerSe", model_name, f'Label{label_name}')
    cktp_files = [file for file in os.listdir(save_folder) if file.endswith(".ckpt")]
    # cktp_files = [file for file in os.listdir(join(save_folder, split_name)) if 'validation_auroc' in file and file.endswith(".ckpt")]
    if len(cktp_files) == 1:
        model_path = os.path.join(save_folder, cktp_files[0])
    else:
        print(f"Error: There is not exactly one .ckpt file in the folder, using the {cktp_files[0]} file")
        model_path = os.path.join(save_folder, cktp_files[0])
    # save_folder = os.path.join(save_folder, fold_no)
    model = PLmodel.load_from_checkpoint(model_path, class_ratio=class_weight, spatial_dims=spatial_dims,
                                         learning_rate=1e-5, network=net,
                                         save_folder=save_folder, inference_only=True, df=df, split='Dataset', fold='Dataset',
                                         label_name=label_name, fracture_timeline=fracture_time,
                                         label_suffix=save_label_suffix)
    trainer = pl.Trainer(devices=1, max_epochs=50)
    trainer.test(model=model, datamodule=dataset_loader)


def inspect_data_loader(image_folder, labels_file, split_name, label_name):

    config_file = DataModuleConfig(num_cpus=2, simlr=False)
    config_file.input_dimension = 2
    config_file.labels_file = labels_file
    config_file.img_dir = image_folder
    config_file.split_name = split_name
    config_file.label = label_name
    dataset_loader = DataModule(config_file)
    dataset_loader.prepare_data()

    test_loader = iter(dataset_loader.train_dataloader())

    for i in range(1):
        images,  labels = next(test_loader)
        # fig, ax = plt.subplots(1, len(images), figsize=(20, 5))
        for idx, im in enumerate(images):
            print(im.shape)
        #     ax[idx].imshow(im.numpy()[0].squeeze(), cmap='gray')
        #     ax[idx].grid(False)
        #     ax[idx].set_xticks([])
        #     ax[idx].set_yticks([])
        # plt.axis('off')
        # plt.grid(b=None)
        # plt.show()

    # im, label = next(test_loader)
    # print(im.shape)
    # print(label.shape)


def main_prog(seed):
    set_seed(seed)
    epoch = 30
    pretrained = True
    for lr in [1e-3]:  #to test for different Learning rates
        for net in ["seresnetbyol100"]:  #to test different models
            for label_name in ["0vs23"]:  #to test different labels : 0vs23, 01vs23, 0vs123
                model_name = f"20260329_{net}_3D_PrevalentFractures_{label_name}_605040_AdamWFixedLR{lr}_outerloop_{seed}"
                start_training(model_name, net, epoch, lr, pretrained)


if __name__ == "__main__":
    # test_network()
    main_prog(898562)
    # image_folder = r"verse19/full_dataset_patches"
    # labels_file = r"verse19/verse19_0vs23_splitted.csv"
    # split_name = 'Dataset'
    # label_name = "0vs23"
    # inspect_data_loader(image_folder, labels_file, split_name, label_name)
    #
    # inspect_data_loader(image_folder=r"MrOs_dataset/patches_arbitary_sized",
    #                     labels_file=rf"MrOs_dataset/MrOS_labels_v8/MrOs_Label_2024_FAANYSPN_SQ1_10years_splitv8_testsplit0.csv",
    #                     split_name='Split4', label_name='IF10SQ1')
