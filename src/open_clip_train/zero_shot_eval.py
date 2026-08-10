"""Zero-shot disease classification evaluation for O-MAKE.

Each benchmark is declared once in ``EVAL_TASKS`` below: the CLI flag that
enables it, the class-name list used to build the text classifier, and the
metric reported. Only the classifiers for the benchmarks you actually request
are built, so evaluating a single dataset does not pay for all thirteen.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, OPENAI_SKIN_TEMPLATES
from open_clip import (
    PAD_CLASSNAMES,
    F17K_DISEASE_113_CLASSES,
    SNU_134_CLASSNAMES,
    SD_128_CLASSNAMES,
    DAFFODIL_5_CLASSNAMES,
    SD_128_tails_10_CLASSNAMES,
    SD_128_tails_30_CLASSNAMES,
    SD_128_tails_50_CLASSNAMES,
    SD_tails_CLASSNAMES,
    SNU_tails_CLASSNAMES,
)
from open_clip import (
    PAD_CLASSNAMES_ONTOLOGY_AUG,
    F17K_DISEASE_113_CLASSES_ONTOLOGY_AUG,
    SNU_134_CLASSNAMES_ONTOLOGY_AUG,
    SD_128_CLASSNAMES_ONTOLOGY_AUG,
    DAFFODIL_5_CLASSNAMES_ONTOLOGY_AUG,
)
from open_clip_train.precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, num_class, args, metric='f1'):
    """Score one benchmark. Returns (primary, secondary, per-sample dataframe)."""
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        true_labels_list = []
        prediction_labels_list = []
        targets_one_hot = []
        predictions_probs = []

        for images, target in tqdm(dataloader):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)
            true_labels = target.to(torch.int64)

            with autocast():
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100.0 * image_features @ classifier
                prediction_softmax = torch.softmax(logits, dim=1)
                prediction_decode = prediction_softmax.argmax(dim=1)

            true_labels_list.extend(true_labels.cpu().numpy())
            prediction_labels_list.extend(prediction_decode.cpu().numpy())
            targets_one_hot.extend(F.one_hot(true_labels, num_classes=num_class).cpu().numpy())
            predictions_probs.extend(prediction_softmax.cpu().numpy())

        true_labels_array = np.array(true_labels_list)
        prediction_labels_array = np.array(prediction_labels_list)
        targets_array = np.array(targets_one_hot)
        predictions_array = np.array(predictions_probs)
        df = pd.DataFrame(data={'pred': prediction_labels_array, 'gt': true_labels_array})

        if metric == 'acc':
            top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
            top5_acc = top_k_accuracy_score(true_labels_array, predictions_array, k=5)
            return top1_acc, float(top5_acc), df

        if metric == 'acc3':
            top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
            top3_acc = top_k_accuracy_score(true_labels_array, predictions_array, k=3)
            return top1_acc, float(top3_acc), df

        if metric == 'f1+acc':
            top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
            wf1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')
            return top1_acc, wf1, df

        if metric == 'auroc+acc':
            auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
            acc = accuracy_score(true_labels_array, prediction_labels_array)
            return auroc, acc, df

        if metric == 'f1':
            auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
            f1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')
            return auroc, f1, df

        raise ValueError(f'unknown metric: {metric}')


@dataclass(frozen=True)
class EvalTask:
    name: str                       # CLI flag, e.g. "pad" -> --eval-pad
    data_key: str                   # key produced by get_zeroshot_eval_data()
    classnames: Sequence[str]       # text classifier vocabulary
    metric: str                     # one of the metrics handled by run()
    result_keys: Sequence[str]      # names of the two returned numbers
    ontology_aug: Optional[Sequence[str]] = None  # variant used with --ontology_aug
    help: str = ''

    def classnames_for(self, args):
        if getattr(args, 'ontology_aug', False) and self.ontology_aug is not None:
            return self.ontology_aug
        return self.classnames


EVAL_TASKS = (
    EvalTask('pad', 'zeroshot_pad', PAD_CLASSNAMES, 'auroc+acc',
             ('zeroshot-pad-auroc', 'zeroshot-pad-acc'),
             PAD_CLASSNAMES_ONTOLOGY_AUG, 'PAD-UFES-20, 6 classes'),
    EvalTask('f17k', 'zeroshot_f17k_113_disease', F17K_DISEASE_113_CLASSES, 'acc',
             ('zeroshot-f17k-113-top1-acc', 'zeroshot-f17k-113-top5-acc'),
             F17K_DISEASE_113_CLASSES_ONTOLOGY_AUG, 'Fitzpatrick17K, 113 classes'),
    EvalTask('snu', 'zeroshot_snu_134', SNU_134_CLASSNAMES, 'acc',
             ('zeroshot-SNU-134-top1-acc', 'zeroshot-SNU-134-top5-acc'),
             SNU_134_CLASSNAMES_ONTOLOGY_AUG, 'SNU, 134 classes'),
    EvalTask('sd128', 'zeroshot_sd_128', SD_128_CLASSNAMES, 'acc',
             ('zeroshot-SD-128-top1-acc', 'zeroshot-SD-128-top5-acc'),
             SD_128_CLASSNAMES_ONTOLOGY_AUG, 'SD-128, 128 classes'),
    EvalTask('daffodil', 'zeroshot_daffodil_5', DAFFODIL_5_CLASSNAMES, 'acc',
             ('zeroshot-Daffodil-5-top1-acc', 'zeroshot-Daffodil-5-top5-acc'),
             DAFFODIL_5_CLASSNAMES_ONTOLOGY_AUG, 'Daffodil, 5 classes'),
    EvalTask('sd128-tail10', 'zeroshot_sd128_tail10', SD_128_tails_10_CLASSNAMES, 'acc',
             ('zeroshot-SD-128(10% tail)-top1-acc', 'zeroshot-SD-128(10% tail)-top5-acc'),
             None, 'SD-128 rarest 10 percent of classes (26)'),
    EvalTask('sd128-tail30', 'zeroshot_sd128_tail30', SD_128_tails_30_CLASSNAMES, 'acc',
             ('zeroshot-SD-128(30% tail)-top1-acc', 'zeroshot-SD-128(30% tail)-top5-acc'),
             None, 'SD-128 rarest 30 percent of classes (60)'),
    EvalTask('sd128-tail50', 'zeroshot_sd128_tail50', SD_128_tails_50_CLASSNAMES, 'acc',
             ('zeroshot-SD-128(50% tail)-top1-acc', 'zeroshot-SD-128(50% tail)-top5-acc'),
             None, 'SD-128 rarest 50 percent of classes (82)'),
    EvalTask('sd-tails', 'zeroshot_sd_tails', SD_tails_CLASSNAMES, 'acc',
             ('zeroshot-SD-tails-top1-acc', 'zeroshot-SD-tails-top5-acc'),
             None, 'SD-198 classes absent from SD-128 (70)'),
    EvalTask('snu-tails', 'zeroshot_snu_tails', SNU_tails_CLASSNAMES, 'acc',
             ('zeroshot-SNU-tails-top1-acc', 'zeroshot-SNU-tails-top5-acc'),
             None, 'SNU classes with <15 samples (85)'),
)

EVAL_TASKS_BY_NAME = {t.name: t for t in EVAL_TASKS}


def task_arg(name: str) -> str:
    """CLI flag --eval-sd-tails -> args attribute eval_sd_tails."""
    return 'eval_' + name.replace('-', '_')


def requested_tasks(args) -> list:
    return [t for t in EVAL_TASKS if getattr(args, task_arg(t.name), None)]


def zero_shot_eval(model, data, args, tokenizer=None):
    """Evaluate every benchmark whose --eval-<name> flag was given."""
    tasks = requested_tasks(args)
    if not tasks:
        logging.warning('No --eval-<dataset> flag was given; nothing to evaluate.')
        return {}

    if args.distributed and not args.horovod:
        model = model.module
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)

    logging.info('Building zero-shot classifiers for: %s', ', '.join(t.name for t in tasks))
    classifiers = {}
    with autocast():
        for task in tasks:
            classifiers[task.name] = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=task.classnames_for(args),
                templates=OPENAI_SKIN_TEMPLATES,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

    results = {}
    for task in tasks:
        logging.info('Evaluating %s (%s)', task.name, task.help)
        primary, secondary, _ = run(
            model,
            classifiers[task.name],
            data[task.data_key].dataloader,
            len(task.classnames_for(args)),
            args,
            metric=task.metric,
        )
        results[task.result_keys[0]] = primary
        results[task.result_keys[1]] = secondary
    return results
