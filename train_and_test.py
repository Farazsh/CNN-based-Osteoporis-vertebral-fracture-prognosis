import monai.networks.nets
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from configs.config import Config
from src.datasets.custom_dataloaders_MrOS import DataModuleMrOS
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
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['figure.dpi'] = 200
optimizers = {"adam": Adam, "sgd": SGD, "adamw": AdamW}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class BaseModel(pl.LightningModule):
    def __init__(self, class_ratio, spatial_dims, learning_rate, network, optimizer_name, threshold, threshold_tuning,
                 save_folder, inference_only, df, split, fold, label_name, fracture_timeline, label_suffix,
                 evaluate_ckpt, epoch):
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

        # Inference only
        self.inference_only = inference_only
        self.label_name = label_name
        self.fracture_timeline = fracture_timeline
        self.suffix = label_suffix
        self.evaluate_ckpt = evaluate_ckpt

        # For recording model outputs for each datapoint
        if inference_only and df is not None:
            self.model_outputs = ['predictions', 'actuals', 'id']
            self.df = df
            self.split = split
            self.fold = fold
            self.df = self.df[self.df[f"Split{self.split}"]=='Testing']
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
            elif self.network.lower() == 'resnet18':
                self.backbone = Resnet18_3d()
            elif self.network.lower() == 'resnet50':
                self.backbone = resnet50_3d()
            elif self.network.lower() == 'densenet201':
                self.backbone = monai.networks.nets.DenseNet201(spatial_dims=3, in_channels=1, out_channels=1)
            else:
                print('Not Implemented', self.network)


        if spatial_dims == 2:
            if self.network.lower() == 'fnet':
                self.backbone = Fnet2D()
            elif self.network.lower() == 'seresnext50':
                self.backbone = monai.networks.nets.SEResNext50(spatial_dims=2, in_channels=1, num_classes=1)
            elif self.network.lower() == 'resnet18':
                self.backbone = Resnet18_2d()
            elif self.network.lower() == 'resnet50':
                self.backbone = resnet50_2d()
            elif self.network.lower() == 'densenet201':
                self.backbone = monai.networks.nets.DenseNet201(spatial_dims=2, in_channels=1, out_channels=1)
            elif self.network.lower() == 'convnext':
                self.backbone = torchvision.models.convnext.convnext_small(
                    weights=torchvision.models.get_model_weights("convnext_small"))
                self.backbone.classifier[2] = nn.Linear(768, 1, bias=True)
            if self.network.lower() == 'convnextbase':
                self.backbone = torchvision.models.convnext.convnext_base(
                    weights=torchvision.models.get_model_weights("convnext_base"))
                self.backbone.classifier[2] = nn.Linear(1024, 1, bias=True)
            else:
                print('Not Implemented', self.network)

        if self.perform_swa:
            self.avg_model = deepcopy(self.backbone)


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

    def xup_loss(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a.float()) + (1 - lam) * self.loss_fn(pred, y_b.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def configure_optimizers(self):
        # Fixed LR
        return self.optimizer(self.parameters(), lr=self.lr)

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
            # todo: read this external file through config
            df = pd.read_excel(r"data/checkpoints/06082024_MrOS_model_results.xlsx", index_col=None)
            data = self.save_folder.split('/')[3].split('_')
            meta_data_length = len(df.columns) - 9  # 7 is number of metrics, 2 for split and epochs
            if len(data) > meta_data_length:
                print('Inconsistent model name, cropping the ends')
                data = data[:meta_data_length]
            elif len(data) < meta_data_length:
                print('Missing variables in model name, appending blanks')
                data.extend(['']*(meta_data_length-len(data)))
            data.extend([self.save_folder.split('/')[4][-1], self.epoch_no])
            data.extend([auroc.item(), auprc.item(), accuracy, precision, recall, f1, specificity])
            df.loc[len(df)] = data
            df.to_excel(r"data/checkpoints/06082024_MrOS_model_results.xlsx", index=False)

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
            prediction, target, id = self.test_metric_dict['predictions'].detach(), \
                                     self.test_metric_dict['actuals'].detach(), \
                                     self.test_metric_dict['id'].detach()
            self.compute_inference(prediction, target, id)
            return

        prediction, target = self.test_metric_dict['predictions'].detach(), self.test_metric_dict['actuals'].detach()
        # _ = plot_auroc(target.flatten().numpy(), prediction.flatten().numpy(), f'Testing_Epoch_{self.epoch_no}',
        #                self.threshold, self.save_folder)
        # plot_auroc_auprc(target.flatten().numpy(), prediction.flatten().numpy(), self.save_folder, self.split, '_Test')
        self.compute_metrics(prediction, target, "test")
        self.test_metric_dict['predictions'] = torch.empty(1, 0)
        self.test_metric_dict['actuals'] = torch.empty(1, 0)

    def compute_inference(self, prediction, target, id) -> None:
        pred_binary = (prediction >= self.threshold).long().flatten().numpy()
        prediction = prediction.flatten().numpy()
        target = target.flatten().numpy()
        id = id.flatten().numpy().astype(int)
        mapper_sigmoid = dict(zip(id, prediction))

        if self.evaluate_ckpt:
            self.df[f"Sigmoid_Epoch{self.epoch_no}"] = self.df.ID.apply(lambda x: mapper_sigmoid[x])
            self.df.to_csv(join(self.save_folder, f"Checkpoint results {self.suffix}.csv"),
                           index=False, sep=';')
            return

        plot_auroc_auprc(target, prediction, self.save_folder, self.split, self.suffix)
        mapper_binary = dict(zip(id, pred_binary))
        self.df[f"Sigmoid_split{self.split}"] = self.df.ID.apply(lambda x: mapper_sigmoid[x])
        self.df[f"Binary_split{self.split}"] = self.df.ID.apply(lambda x: mapper_binary[x])
        self.df.to_csv(join(self.save_folder, f"{self.network}_{self.spatial_dims}D_sigmoid_results_fold_{self.fold}{self.suffix}.csv"),
                       index=False, sep=';')
        # box_strip_plot(self.df, self.split, self.label_name, self.save_folder)
        plot_sigmoidVsFractureTime(self.df, self.split, self.save_folder, self.threshold, self.fracture_timeline, self.suffix)


class PLmodel(BaseModel):
    def __init__(self, class_ratio: float, spatial_dims: int, learning_rate: float, network: str = 'fnet',
                 optimizer_name: str = "adamw", threshold: float = 0.5, threshold_tuning: str = False,
                 save_folder: str = r"checkpoints/MrOS/dump", inference_only=False, df=None, split=0, fold=0,
                 label_name='IF10SQ1', fracture_timeline=3653, label_suffix='', evaluate_ckpt=False, epoch=0):

        super(PLmodel, self).__init__(class_ratio, spatial_dims, learning_rate, network, optimizer_name,
                                      threshold, threshold_tuning, save_folder, inference_only, df, split, fold,
                                      label_name, fracture_timeline, label_suffix, evaluate_ckpt, epoch)

    def __str__(self):
        return 'PLmodel'


def start_training(model_name, labels_file, image_folder, monitoring_metric, monitoring_mode, spatial_dims, input_size,
                   label_name, split_nos, network, is_segmented, threshold_tuning, perform_inference, fracture_timeline,
                   max_epoch, fold_no):
    # Parameters
    net_architecture = network
    learning_rate = 5e-5
    label = label_name
    epochs = max_epoch
    batch_size = 32
    spatial_dims = spatial_dims
    optimizer_name = 'adamw'
    is_segmented = is_segmented

    project_name = "MrOS_Prognostic"
    group_name = model_name
    wandb.login(key="64339572cb325d13488524ac730cc32f507bd8ad")
    wandb_logging_config = {
        'net_architecture': net_architecture,
        'spatial_dims': spatial_dims,
        'learning_rate': learning_rate,
        'optimizer': optimizer_name,
        'label': label,
        'epochs': epochs,
        'is_segmented': is_segmented,
        'image_dir': image_folder,
        'labels_filename': labels_file,
        'task_name': project_name,
        'threshold_tuning': threshold_tuning,
        'monitoring_metric': monitoring_metric,
        'monitoring_mode': monitoring_mode,
    }

    checkpoint_folder_main = rf"data/checkpoints/MrOS/{model_name}"
    os.makedirs(checkpoint_folder_main, exist_ok=True)
    with open(join(checkpoint_folder_main, 'training_info.json'), "w") as outfile:
        json.dump(wandb_logging_config, outfile)

    for split_no in split_nos:
        split_name = f"Split{split_no}"
        checkpoint_folder = rf"data/checkpoints/MrOS/{model_name}/{split_name}"
        os.makedirs(checkpoint_folder, exist_ok=True)
        checkpoint_filename = group_name + '_' + split_name + '_top_ckpt_at_{epoch}_with_{validation_auroc:.3f}'
        experiment_name = f"Fold{fold_no}"

        wandb.init(project=project_name, name=experiment_name, config=wandb_logging_config, group=group_name)
        best_checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_folder,
                                                   filename=checkpoint_filename,
                                                   save_top_k=1,
                                                   monitor=monitoring_metric,
                                                   mode=monitoring_mode)

        config_file = Config(num_cpus=2, simlr=False)
        config_file.datamodule.test_as_validation = True
        config_file.datamodule.labels_file = labels_file
        config_file.datamodule.img_dir = image_folder
        config_file.datamodule.split_name = split_name
        config_file.datamodule.label = label_name
        config_file.datamodule.input_size = input_size
        config_file.datamodule.batch_size = batch_size
        config_file.datamodule.input_dimension = spatial_dims
        sample_dataset_loader = DataModuleMrOS(config_file)
        sample_dataset_loader.prepare_data()
        class_ratio = sample_dataset_loader.get_class_ratio()

        model = PLmodel(class_ratio=class_ratio, spatial_dims=spatial_dims, learning_rate=learning_rate,
                        network=net_architecture, optimizer_name=optimizer_name, fold=fold_no,
                        threshold_tuning=threshold_tuning, save_folder=checkpoint_folder)
        wandb_logger = WandbLogger(project=project_name, name=experiment_name)
        trainer = pl.Trainer(devices=1, max_epochs=epochs, logger=wandb_logger,
                             # limit_train_batches=0.66,
                             callbacks=[
                                 best_checkpoint_callback,
                                 EarlyStopping(monitor=monitoring_metric, min_delta=0.01, patience=5, verbose=False,
                                               mode=monitoring_mode)
                                 ]
                             )
        trainer.fit(model=model, datamodule=sample_dataset_loader)
        # trainer.save_checkpoint(join(checkpoint_folder,group_name+'_'+split_name+f'Model_trained_for_epochs_{epochs}.ckpt'))

        best_model_path = glob.glob(join(checkpoint_folder, '*.ckpt'))[0]
        match = re.search(r"epoch=(\d+)", best_model_path)
        if match:
            epoch_number = match.group(1)
        else:
            epoch_number = 0
        save_folder = f"/data/checkpoints/Censored_sub_evaluation"
        model = PLmodel.load_from_checkpoint(best_model_path, class_ratio=class_ratio, spatial_dims=spatial_dims,
                                             network=net_architecture, learning_rate=learning_rate, threshold=0.5,
                                             fold=fold_no, save_folder=checkpoint_folder, split=split_no, epoch=epoch_number)
        trainer.test(model=model, datamodule=sample_dataset_loader)

        # Incase of model training with 4-fold cv, the data loader is re-initialized with test data as is
        config_file = Config(num_cpus=2)
        config_file.datamodule.labels_file = labels_file
        config_file.datamodule.img_dir = image_folder
        config_file.datamodule.split_name = split_name
        config_file.datamodule.label = label_name
        config_file.datamodule.input_size = input_size
        config_file.datamodule.batch_size = batch_size
        config_file.datamodule.input_dimension = spatial_dims
        sample_dataset_loader = DataModuleMrOS(config_file)
        sample_dataset_loader.test_as_validation = False
        sample_dataset_loader.prepare_data()
        trainer.test(model=model, datamodule=sample_dataset_loader)

        if perform_inference:
            df = pd.read_csv(labels_file, delimiter=';', index_col=None)
            # best_model.change_model_state_to_inference(df, split_no, label_name, fracture_timeline)
            model.change_model_state_to_inference(df, split_no, fold_no, label_name, fracture_timeline)
            config_file.datamodule.Inference_only = True
            sample_dataset_loader = DataModuleMrOS(config_file)
            sample_dataset_loader.prepare_data()
            # trainer.test(model=best_model, datamodule=sample_dataset_loader)
            trainer.test(model=model, datamodule=sample_dataset_loader)

        wandb.finish()


def test_network():
    image_batch = torch.tensor(np.random.rand(16, 3, 50, 40), dtype=torch.float32)
    label = torch.randint(0, 1, (16,))

    # backbone = SEResNext50(spatial_dims=2, in_channels=1, num_classes=1)

    weights_enum = torchvision.models.get_model_weights("convnext_base")
    backbone = torchvision.models.convnext.convnext_base(weights=weights_enum)
    # print(backbone)
    backbone.classifier[2] = nn.Linear(1024, 1)  # for convnext base
    # backbone.classifier[2] = nn.Linear(1536, 1)  # for convnext large
    out = backbone(image_batch)
    #
    count_and_print_num_of_parameters(backbone)
    print("Output Shape: ", out.shape)
    print("Labels Shape: ", label.shape)


def preprocess_and_inference(save_folder, csv_filename, image_dir, split_no, fold_no, spatial_dims, net,
                             label_name, fracture_time, train_as_test=False, validation_as_test=False):
    df = pd.read_csv(csv_filename, delimiter=';', index_col=None)
    split_name = f'Split{split_no}'
    config_file = Config(num_cpus=2, simlr=False)
    config_file.datamodule.labels_file = csv_filename
    config_file.datamodule.img_dir = image_dir
    config_file.datamodule.split_name = split_name
    config_file.datamodule.label = label_name
    config_file.datamodule.Inference_only = True
    config_file.datamodule.input_dimension = spatial_dims
    sample_dataset_loader = DataModuleMrOS(config_file)
    sample_dataset_loader.prepare_data()
    class_weight = sample_dataset_loader.get_class_ratio()

    if train_as_test and validation_as_test:
        print("Cannot perform inference on train and validation set at once")
        return
    if train_as_test:
        sample_dataset_loader.train_as_test = True
        df[split_name] = df[split_name].replace({'Training': 'Testing', 'Testing': 'Training'})
        save_label_suffix = '_Train'
    elif validation_as_test:
        sample_dataset_loader.validation_as_test = True
        df[split_name] = df[split_name].replace({'Validation': 'Testing', 'Testing': 'Validation'})
        save_label_suffix = '_Validation'
    else:
        save_label_suffix = '_Test'

    cktp_files = [file for file in os.listdir(join(save_folder, split_name)) if file.endswith(".ckpt")]
    # cktp_files = [file for file in os.listdir(join(save_folder, split_name)) if 'validation_auroc' in file and file.endswith(".ckpt")]
    if len(cktp_files) == 1:
        model_path = os.path.join(save_folder, split_name, cktp_files[0])
    else:
        print(f"Error: There is not exactly one .ckpt file in the folder, using the {cktp_files[0]} file")
        model_path = os.path.join(save_folder, split_name, 'epoch_15', cktp_files[0])
    save_folder = f"/data/checkpoints/saved_inferences"
    model = PLmodel.load_from_checkpoint(model_path, class_ratio=class_weight, spatial_dims=spatial_dims,
                                         learning_rate=1e-5, network=net,
                                         save_folder=save_folder, inference_only=True, df=df, split=split_no, fold=fold_no,
                                         label_name=label_name, fracture_timeline=fracture_time,
                                         label_suffix=save_label_suffix)
    trainer = pl.Trainer(devices=1, max_epochs=50)
    trainer.test(model=model, datamodule=sample_dataset_loader)


def inspect_data_loader(image_folder, labels_file, split_name, label_name):

    config_file = Config(num_cpus=2, simlr=False)
    config_file.datamodule.input_dimension = 2
    config_file.datamodule.labels_file = labels_file
    config_file.datamodule.img_dir = image_folder
    config_file.datamodule.split_name = split_name
    config_file.datamodule.label = label_name
    sample_dataset_loader = DataModuleMrOS(config_file)
    sample_dataset_loader.prepare_data()

    test_loader = iter(sample_dataset_loader.train_dataloader())

    for i in range(1):
        images,  labels = next(test_loader)
        fig, ax = plt.subplots(1, len(images), figsize=(20, 5))
        for idx, im in enumerate(images):
            ax[idx].imshow(im.numpy()[0].squeeze(), cmap='gray')
            ax[idx].grid(False)
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
        plt.axis('off')
        plt.grid(b=None)
        plt.show()


def main():
    version = 1
    label = 'FAANYSPN'  # Name of the column in the MrOS database file "Fracture analysis 2023", indicates any fracture
    input_size = 47  # Size of input image
    network = 'convnextbase'  # Alias for the network architecture
    spatial_dims = 2  # Whether to use 2D or 3D
    image_folder_name_3d = r"MrOs_dataset/patches_arbitary_sized"  # Folder containing nii files of 3D vertebral patches
    max_epoch = 50
    split_list = [4]

    for fold in [0, 1, 2, 3]:  # 4 splits (subsets) of the dataset, each used as test set once
        data_file = rf"MrOs_dataset/MrOS_labels_v8/MrOs_Label_2024_FAANYSPN_SQ1_10years_splitv8_testsplit{fold}.csv"
        model_name = f"06082024_{network}_{spatial_dims}D_IF10SQ1_{label}_{str(input_size)*3}_NoSegBS32_AdamwFixedLR5e6_outerloop_{fold}_v{version}"
        start_training(model_name, data_file, image_folder_name_3d, 'validation_loss', 'min', spatial_dims, input_size,
                       f'IF10SQ1', split_list, network, False, False, False, 3653, max_epoch, fold)


if __name__ == "__main__":
    main()
