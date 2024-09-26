import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
import sklearn


class SELayer(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        out_channel = int(in_channel/reduction)
        self.fc = nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_channel, in_channel, bias=False),
                nn.Sigmoid(),)
        self.SpatialSqueezeExcitation = nn.Sequential(
                                        nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, bias=False),
                                        nn.Sigmoid(),)

    def forward(self, x):
        batch, channel, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(batch, channel)
        y = self.fc(y)
        y = y.view(batch, channel, 1, 1, 1)
        cse = x * y.expand_as(x)
        return cse
        # sse = self.SpatialSqueezeExcitation(x)
        # return cse + sse


def plot_auroc(y, y_pred, label, use_threshold=False, save_path=r"checkpoints/MrOS/dump"):
    # thresholds = np.linspace(0, 1, 101)
    tpr, fpr, precision, recall, f1, tpr_minus_fpr, thresholds = calculate_auroc_stats(y, y_pred)
    # fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    if use_threshold:
        best_thresh = use_threshold
        if use_threshold in thresholds:
            ix = np.where(thresholds == use_threshold)[0][
                0]  # return index at given threshold value, index of first if multiple
        else:
            ix = np.argmax(tpr_minus_fpr)
            best_thresh = thresholds[ix]
    else:
        ix = np.argmax(tpr_minus_fpr)
        best_thresh = thresholds[ix]

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.', label=label)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black',
                label=f'best threshold: {best_thresh :.2f} at f1: {f1[ix] :.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_path, label + '_AUROC.png'))
    plt.close()

    plt.plot(recall, precision, marker='.', label=label)
    plt.scatter(recall[ix], precision[ix], marker='o', color='black',
                label=f'best threshold: {best_thresh :.2f} at f1: {f1[ix] :.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_path, label + '_PR.png'))
    plt.close()

    return float(best_thresh)


def calculate_auroc_stats(y, y_pred):
    # thresholds = np.linspace(0, 1, 101)
    thresholds = np.sort(np.unique(np.round(y_pred, 2)))
    if len(thresholds) < 30:
        thresholds = np.linspace(0, 1, 101)
    beta_sq = 4  # Commonly used value of beta is 2, source:wikipedia F1-score
    tpr_list, fpr_list, precision_list, recall_list, f1_list = [], [], [], [], []
    for thresh in thresholds:
        preds = y_pred >= thresh
        tp = np.sum(y * preds)
        tn = np.sum((1 - y) * (1 - preds))
        fp = np.sum((1 - y) * preds)
        fn = np.sum(y * (1 - preds))
        tpr_list.append(tp / (tp + fn))
        fpr_list.append(fp / (fp + tn))
        precision_list.append(tp / (tp + fp))
        recall_list.append(tp / (tp + fn))
        # f1_list.append(((1 + beta_sq)*tp)/((1 + beta_sq)*tp + beta_sq*fp + fn))
        f1_list.append((2 * tp) / (2 * tp + fp + fn))
    tpr_minus_fpr = [x - y for x, y in zip(tpr_list, fpr_list)]

    return tpr_list, fpr_list, precision_list, recall_list, f1_list, tpr_minus_fpr, thresholds


def check_auroc():
    y1 = np.array([0] * 132 + [1] * 8)
    np.random.shuffle(y1)
    y2 = np.random.uniform(0, 1, 140)
    plot_auroc(y1, y2, 'check2')


def count_and_print_num_of_parameters(some_model):
    total_params = 0
    req_grad = 0
    for _, parameter in some_model.named_parameters():
        if parameter.requires_grad:
            req_grad += parameter.numel()
        total_params += parameter.numel()
    print(f"Total Params: {total_params / 1e6 :.2f} M")
    print(f"Total Trainable Params: {req_grad / 1e6 :.2f} M")


def plot_sigmoidVsFractureTime(df, split_no, save_folder, threshold, timeline, suffix):
    plt.rcParams['figure.figsize'] = [7, 5]
    plt.rcParams['figure.dpi'] = 200
    _df = df[df.HasFracture == 1]
    _df.sort_values(['Site'], inplace=True)

    fig, axes = plt.subplots(1, 1)

    # add a bit more breathing room around the axes for the frames
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.2, hspace=0.8)
    fig.patch.set_linewidth(10)
    sns.scatterplot(data=_df, x='TimeToFrax', y=f'Sigmoid_split{split_no}', hue='Site', style='Site', s=100)
    plt.axhline(y=threshold, color='gray', linestyle='--', label='threshold')
    plt.axvline(x=timeline, color='red', linestyle='--', label='timeline')
    plt.ylim((0, 1))
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Sigmoid versus time to fracture')
    plt.tight_layout()
    plt.savefig(join(save_folder, f"Prediction_on_fractured_subjects_split{split_no}{suffix}.png"))
    # plt.show()
    plt.close()


def box_strip_plot(df, split_no, label_name, save_path=r"checkpoints/MrOS/dump"):
    plt.rcParams['figure.figsize'] = [7, 5]
    plt.rcParams['figure.dpi'] = 200
    plt.axhline(y=0.5, c='r', linestyle='--')
    ax = sns.boxplot(data=df, x=label_name, y=f"Sigmoid_split{split_no}",
                     hue=f"Vertebra", hue_order=['L1', 'L2'], boxprops={'alpha': 0.4})
    sns.stripplot(data=df, x=label_name, y=f"Sigmoid_split{split_no}",
                  hue=f"Vertebra", hue_order=['L1', 'L2'], dodge=True, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[(handles[0], handles[2]), (handles[1], handles[3])],
              labels=['L1', 'L2'],
              loc='best', handlelength=4,
              handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.ylim((0, 1))
    plt.xlabel('ground truth labels')
    plt.title(f'model predictions for split {split_no}')
    plt.tight_layout()
    plt.savefig(join(save_path, f'model_prediction_sigmoid_split{split_no}.png'))
    plt.close()
    # plt.show()
    

def plot_auroc_auprc(target, prediction, save_folder, split, suffix):
    
    # Plot AUROC curve
    fpr, tpr, thresh = sklearn.metrics.roc_curve(target, prediction)
    auc = sklearn.metrics.roc_auc_score(target, prediction)
    plt.plot(fpr, tpr, label="auc=" + str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_folder, f"AUROC_curve_Split{split}{suffix}.png"))
    plt.close()

    # Plot AUPRC curve
    precision, recall, _ = sklearn.metrics.precision_recall_curve(target, prediction)
    auprc = sklearn.metrics.average_precision_score(target, prediction)
    plt.plot(recall, precision, label="auprc=" + str(auprc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_folder, f"AUPRC_curve_Split{split}{suffix}.png"))
    plt.close()


def main():
    df = pd.read_csv(r"checkpoints2/Diagnostic_Bilanz/12212023_Fnet3D_PrevalentFractures_Adamw_fixedLR_cw_474747_testfold_2_v2/Split1/Sigmoid results split1.csv", delimiter=';', index_col=None)
    box_strip_plot(df, 1)


if __name__ == "__main__":
    main()
