import math
import sys
import time
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import furnace.utils as utils
from furnace.utils import *
from typing import Iterable, Optional
from pycm import *
import numpy as np

import os
import csv

from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix, top_k_accuracy_score, multilabel_confusion_matrix
from sklearn.utils import resample

import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import torch
import torch.nn.functional as F
import torch.nn as nn

import pandas as pd
from tqdm import tqdm

def misc_measures(confusion_matrix):
    bacc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []

    for i in range(1, confusion_matrix.shape[0]):
        cm1 = confusion_matrix[i]
        bacc.append(1. * (cm1[0, 0] + cm1[1, 1]) / np.sum(cm1))
        sensitivity_ = 1. * cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        sensitivity.append(sensitivity_)
        specificity_ = 1. * cm1[0, 0] / (cm1[0, 1] + cm1[0, 0])
        specificity.append(specificity_)
        precision_ = 1. * cm1[1, 1] / (cm1[1, 1] + cm1[0, 1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_ * specificity_))
        F1_score_2.append(2 * precision_ * sensitivity_ / (precision_ + sensitivity_))
        mcc = (cm1[0, 0] * cm1[1, 1] - cm1[0, 1] * cm1[1, 0]) / np.sqrt(
            (cm1[0, 0] + cm1[0, 1]) * (cm1[0, 0] + cm1[1, 0]) * (cm1[1, 1] + cm1[1, 0]) * (cm1[1, 1] + cm1[0, 1]))
        mcc_.append(mcc)

    bacc = np.array(bacc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()

    return bacc, sensitivity, specificity, precision, G, F1_score_2, mcc_
def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, _, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if "lr_scale" in param_group:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    else:
                        param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(now_time, "Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_predictions_csv(model, data_loader, device, original_csv_path, output_dir, dataset_name):
    # Read the original CSV file
    original_df = pd.read_csv(original_csv_path)

    # Filter for test split only
    test_df = original_df[original_df['split'] == 'test'].copy()

    # Create a list to store predictions
    results = []

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Making predictions"):
            images, filenames, targets = batch[0], batch[1], batch[-1]
            images = images.to(device, non_blocking=True)

            # Compute output
            output = model(images)

            # Get predicted class and probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            _, preds = torch.max(probs, 1)

            # Store results
            for filename, true_label, pred, prob in zip(filenames, targets, preds, probs):
                results.append({
                    'filename': filename,
                    'true_label': true_label.item(),
                    'predicted_label': pred.item(),
                    'probabilities': prob.cpu().numpy()
                })

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write results to CSV
    output_csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ['filename', 'true_label', 'predicted_label'] + [f'probability_class_{i}' for i in
                                                                  range(len(results[0]['probabilities']))]
        writer.writerow(header)

        # Write data
        for result in results:
            row = [
                      result['filename'],
                      result['true_label'],
                      result['predicted_label']
                  ] + [f"{prob:.8f}" for prob in result['probabilities']]
            writer.writerow(row)

    print(f"Predictions for {dataset_name} saved to {output_csv_path}")


@torch.no_grad()
def evaluate(data_loader, model, device, out_dir, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    image_names = []

    # switch to evaluation mode
    model.eval()

    results = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        image_names.extend(batch[1])  # Assuming batch[1] contains image names
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label = F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        prediction_softmax = nn.Softmax(dim=1)(output)
        _, prediction_decode = torch.max(prediction_softmax, 1)
        _, true_label_decode = torch.max(true_label, 1)

        prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
        true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
        true_label_onehot_list.extend(true_label.cpu().detach().numpy())
        prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        # Store results for CSV
        for filename, true_label, pred, prob in zip(batch[1], true_label_decode, prediction_decode, prediction_softmax):
            # print('fffffffffffff',filename)
            results.append({
                'filename': filename,
                'true_label': true_label.item(),
                'predicted_label': pred.item(),
                'probabilities': prob.cpu().numpy()
            })

    # Convert lists to numpy arrays
    true_label_decode_array = np.array(true_label_decode_list)
    prediction_decode_array = np.array(prediction_decode_list)
    true_label_onehot_array = np.array(true_label_onehot_list)
    prediction_array = np.array(prediction_list)

    confusion_matrices = multilabel_confusion_matrix(true_label_decode_array, prediction_decode_array)
    
    sensitivity_list = []
    specificity_list = []

    for idx, cm in enumerate(confusion_matrices):
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
    
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity_list.append(sensitivity)
    
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_list.append(specificity)

    sensitivity = np.array(sensitivity_list)
    specificity = np.array(specificity_list)

    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)

    bacc = balanced_accuracy_score(true_label_decode_array, prediction_decode_array)
    acc = accuracy_score(true_label_decode_array, prediction_decode_array)
    if num_class > 3:
        top3_acc = top_k_accuracy_score(true_label_decode_array, prediction_array, k=3, labels=np.arange(num_class))
        top5_acc = top_k_accuracy_score(true_label_decode_array, prediction_array, k=5, labels=np.arange(num_class))
    else:
        top3_acc = 0
        top5_acc = 0
    auc_roc = roc_auc_score(true_label_onehot_array, prediction_array, multi_class='ovr', average='macro')


    f1 = f1_score(true_label_decode_array, prediction_decode_array, average='weighted')
    recall = recall_score(true_label_decode_array, prediction_decode_array, average='macro')
    recall1 = recall_score(true_label_decode_array, prediction_decode_array, average='weighted')

    print('-------------', mode, '-------------')
    print(
        'Sklearn Metrics - BAcc: {:.4f} Acc: {:.4f} Recall_macro: {:.4f} Recall_weighted: {:.4f} AUC-ROC: {:.4f} Weighted F1-score: {:.4f}'.format(
            bacc, acc, recall, recall1, auc_roc, f1))
    
    # Record with dict for return value
    metrics = {
    'balanced_accuracy': balanced_accuracy_score(true_label_decode_array, prediction_decode_array),
    'accuracy': accuracy_score(true_label_decode_array, prediction_decode_array),
    'top3 accuracy': top3_acc,
    'top5 accuracy': top5_acc,
    'sensitivity': macro_sensitivity,
    'specificity': macro_specificity,
    'auc_roc': roc_auc_score(true_label_onehot_array, prediction_array, multi_class='ovr', average='macro'),
    'weighted_f1': f1_score(true_label_decode_array, prediction_decode_array, average='weighted'),
    'recall_macro': recall_score(true_label_decode_array, prediction_decode_array, average='macro'),
    'recall_weighted': recall_score(true_label_decode_array, prediction_decode_array, average='weighted'),
    }

    if mode == 'test':
        print(metrics)

    # Save results to CSV
    os.makedirs(out_dir, exist_ok=True)
    output_csv_path = os.path.join(out_dir, f"{mode}.csv")
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ['filename', 'true_label', 'predicted_label'] + [f'probability_class_{i}' for i in range(num_class)]
        writer.writerow(header)

        # Write data
        for result in results:
            row = [
                      result['filename'],
                      result['true_label'],
                      result['predicted_label']
                  ] + [f"{prob:.8f}" for prob in result['probabilities']]
            writer.writerow(row)

    print(f"Predictions for {mode} saved to {output_csv_path}")

    # Prepare wandb results
    if mode == 'val':
        wandb_res = {
            'Epoch': epoch,
            'Val Loss': loss.item(),
            'Val BAcc': bacc,
            'Val Acc': acc,
            'Val ROC': auc_roc,
            f'{mode.capitalize()} W_F1': f1,
            f'{mode.capitalize()} Recall_macro': recall,
            f'{mode.capitalize()} Recall_weighted': recall1,
        }
    else:
        wandb_res = {
            f'{mode.capitalize()} BAcc': bacc,
            f'{mode.capitalize()} Acc': acc,
            f'{mode.capitalize()} ROC': auc_roc,
            f'{mode.capitalize()} F1': f1,
            f'{mode.capitalize()} Recall_macro': recall,
            f'{mode.capitalize()} Recall_weighted': recall1
        }

    return metrics, wandb_res

from torchvision.transforms.functional import to_pil_image, to_tensor

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_inference_transforms():
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])

def batch_tta_transforms(batch_tensor, num_augmentations=5):
    device = batch_tensor.device
    batch_np = batch_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert batch from BxCxHxW to BxHxWxC
    transforms = get_inference_transforms()
    augmented_batches = []

    for _ in range(num_augmentations):
        augmented_images = [transforms(image=img_np)['image'] for img_np in batch_np]
        augmented_images = np.stack(augmented_images)  # Stack along a new dimension
        augmented_batches.append(torch.from_numpy(augmented_images).to(device))

    # Stack along the augmentation dimension to form [num_augmentations, batch_size, channels, height, width]
    return torch.stack(augmented_batches, dim=0)

class TTAHandler:
    def __init__(self, num_augmentations=5):
        self.num_augmentations = num_augmentations
        self.transforms = self.get_inference_transforms()

    def get_inference_transforms(self):
        return transforms.Compose([
            transforms.Resize(246),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=(0, 270))], p=0.3),
            transforms.RandomApply([transforms.ColorJitter(hue=0.25, saturation=0.25)], p=0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def apply_transforms(self, batch_tensor):
        device = batch_tensor.device
        augmented_batches = []

        for _ in range(self.num_augmentations):
            augmented_images = [self.transforms(img) for img in batch_tensor]
            augmented_batches.append(torch.stack(augmented_images, dim=0))

        return torch.stack(augmented_batches, dim=0)

@torch.no_grad()
def evaluate_tta(data_loader, model, device, out_dir, epoch, mode, num_class, num_TTA=5):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    tta_handler = TTAHandler(num_augmentations=num_TTA)
    results = []
    all_preds = []
    all_targets = []
    logits = []
    for batch in metric_logger.log_every(data_loader, 10, 'Test:'):
        images, targets = batch[0].to(device), batch[-1].to(device)
        batch_image_names = batch[1]
        batch_image_list = []
        batch_image_list.extend(batch_image_names)

        tta_images = tta_handler.apply_transforms(images).to(device)  # TTA transform  [TTA_num, B, H, W]
        _, B,_, _, _ = tta_images.shape
        tta_inputs = tta_images.view(-1, *tta_images.shape[2:]) #  [TTA_num * B, H, W]
        outputs = model(tta_inputs)  # Reshape for model input
        reshaped_outputs = outputs.reshape(num_TTA, B, -1)

        predictions = torch.mean(reshaped_outputs, dim=0)

        _, predicted = torch.max(predictions, 1)
        logits.extend(predictions.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        probabilities = torch.softmax(predictions, dim=1).cpu().numpy()
        
        for i, image_name in enumerate(batch_image_list):
            results.append({
                'filename': image_name,
                'true_label': targets[i].item(),
                'predicted_label': predicted[i].item(),
                'probabilities': probabilities[i]
            })

    # CM
    # cm = confusion_matrix(np.array(all_targets), np.array(all_preds))
    confusion_matrices = multilabel_confusion_matrix(np.array(all_targets), np.array(all_preds))
    sensitivity_list = []
    specificity_list = []
    
    for idx, cm in enumerate(confusion_matrices):
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
    
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        sensitivity_list.append(sensitivity)
    
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_list.append(specificity)

    sensitivity = np.array(sensitivity_list)
    specificity = np.array(specificity_list)

    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    
    # print("Confusion Matrix:")
    # print(confusion_matrices)
    probabilities = np.array(logits)

    metrics = {
        'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        'accuracy': accuracy_score(all_targets, all_preds),
        # 'auc_roc': roc_auc_score(F.one_hot(torch.tensor(all_targets), num_classes=num_class), np.array([r['probabilities'] for r in results]), multi_class='ovr'),
        'top3_acc' : top_k_accuracy_score(all_targets, probabilities, k=3, labels=np.arange(num_class)),
        'top5_acc' : top_k_accuracy_score(all_targets, probabilities, k=5, labels=np.arange(num_class)),
        'sensitivity': macro_sensitivity,
        'specificity': macro_specificity,
        'weighted_f1': f1_score(all_targets, all_preds, average='weighted'),
        'recall_macro': recall_score(all_targets, all_preds, average='macro'),
        'recall_weighted': recall_score(all_targets, all_preds, average='weighted'),
    }
    print(metrics)

    # Save results to CSV
    # os.mkdir(out_dir, exist_ok=True)
    output_csv_path = os.path.join(out_dir, f"{mode}.csv")
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ['filename', 'true_label', 'predicted_label'] + [f'probability_class_{i}' for i in range(num_class)]
        writer.writerow(header)

        # Write data
        for result in results:
            row = [
                      result['filename'],
                      result['true_label'],
                      result['predicted_label']
                  ] + [f"{prob:.8f}" for prob in result['probabilities']]
            writer.writerow(row)

    print(f"Predictions for {mode} saved to {output_csv_path}")
    return metrics, None