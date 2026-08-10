import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip_train.precision import get_autocast

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    OPENAI_SKIN_TEMPLATES,HAM_CLASSNAMES,PAD_CLASSNAMES, DERMNET_CLASSNAMES, F17K_DISEASE_82_CLASSES, F17K_DISEASE_9_CLASSES, F17K_DISEASE_9_CLASSES, HAM_2_CLASSNAMES, SKINCON_32_CLASSES, DDI_CLASSNAMES, BCN_CLASSNAMES, \
    SNU_134_CLASSNAMES, SCIN_20_CLASSNAMES, SCIN_211_CLASSNAMES, SD_128_CLASSNAMES, DAFFODIL_5_CLASSNAMES, F17K_DISEASE_113_CLASSES

from open_clip import SD_128_CLASSNAMES_ONTOLOGY_AUG, SNU_134_CLASSNAMES_ONTOLOGY_AUG, DAFFODIL_5_CLASSNAMES_ONTOLOGY_AUG, F17K_DISEASE_113_CLASSES_ONTOLOGY_AUG, PAD_CLASSNAMES_ONTOLOGY_AUG

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, top_k_accuracy_score
import numpy as np
from torch import nn


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, num_class, args, skincon=False, metric='f1', task=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        if skincon:
            # Multi-label classification
            targets = []
            predictions = []
            for images, target in tqdm(dataloader, unit_scale=args.batch_size):
                images = images.to(device=device, dtype=input_dtype)
                # Convert string targets to list of integers
                target = [[int(element) for element in i.strip('[]').split()] for i in target]
                target = torch.Tensor(target).to(device)
                true_labels = target  # Multi-label targets

                with autocast():
                    output = model(image=images)
                    image_features = output['image_features'] if isinstance(output, dict) else output[0]
                    logits = 100.0 * image_features @ classifier

                targets.extend(true_labels.cpu().numpy())
                predictions.extend(logits.cpu().numpy())

            # Compute AUROC for multi-label
            targets_array = np.array(targets)
            predictions_array = np.array(predictions)
            auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
            # Binarize predictions for F1 score
            prediction_labels = (predictions_array >= 0.5).astype(int)
            f1 = f1_score(targets_array, prediction_labels, average='weighted')

            return auroc, f1

        else:
            # Multi-class classification
            top1_correct = 0.0
            total_samples = 0.0
            true_labels_list = []
            prediction_labels_list = []
            targets_one_hot = []
            predictions_probs = []

            for images, target in tqdm(dataloader, unit_scale=args.batch_size):
                images = images.to(device=device, dtype=input_dtype)
                target = target.to(device)
                true_labels = target.to(torch.int64)

                with autocast():
                    output = model(image=images, infer=True)
                    image_features = output['image_features'] if isinstance(output, dict) else output[0]
                    logits = 100.0 * image_features @ classifier
                    prediction_softmax = torch.softmax(logits, dim=1)
                    prediction_decode = prediction_softmax.argmax(dim=1)

                # Compute accuracy
                acc1 = accuracy(logits, true_labels, topk=(1,))
                batch_size = images.size(0)
                top1_correct += acc1[0] * batch_size / 100.0  # Convert percentage back to count
                total_samples += batch_size

                # Collect data for metrics
                true_labels_list.extend(true_labels.cpu().numpy())
                prediction_labels_list.extend(prediction_decode.cpu().numpy())
                targets_one_hot.extend(F.one_hot(true_labels, num_classes=num_class).cpu().numpy())
                predictions_probs.extend(prediction_softmax.cpu().numpy())

            # Compute metrics
            top1_acc = top1_correct / total_samples * 100.0  # Convert back to percentage
            true_labels_array = np.array(true_labels_list)
            prediction_labels_array = np.array(prediction_labels_list)
            targets_array = np.array(targets_one_hot)
            predictions_array = np.array(predictions_probs)

            if task == 'SNU' or task == 'HAM':
                np.save(f'/{task}_targets_array.npy', targets_array)
                np.save(f'/{task}_predictions_array.npy', predictions_array)

            if metric == 'f1':
                auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
                f1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')

                return auroc, f1
            elif metric == 'acc':
                top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
                top5_acc = top_k_accuracy_score(true_labels_array, predictions_array, k=5)

                return top1_acc, top5_acc

            elif metric == 'f1+acc':
                top1_acc = accuracy_score(true_labels_array, prediction_labels_array)
                wf1 = f1_score(true_labels_array, prediction_labels_array, average='weighted')
                return top1_acc, wf1
            elif metric == 'auroc+acc':
                auroc = roc_auc_score(targets_array, predictions_array, multi_class='ovr', average='macro')
                acc = accuracy_score(true_labels_array, prediction_labels_array)
                return auroc, acc


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    # if 'imagenet-val' not in data and 'imagenet-v2' not in data:
    #     return {}
    print('data source:',data)
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)

    templates=OPENAI_SKIN_TEMPLATES
    with autocast():
        classifier_dermnet = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=DERMNET_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_pad = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=PAD_CLASSNAMES_ONTOLOGY_AUG if args.ontology_aug else PAD_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_ham = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=HAM_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_f17k_113_disease = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=F17K_DISEASE_113_CLASSES_ONTOLOGY_AUG if args.ontology_aug else F17K_DISEASE_113_CLASSES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_f17k_9_disease = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=F17K_DISEASE_9_CLASSES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_skincon_32_disease = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=SKINCON_32_CLASSES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_ham_2 = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=HAM_2_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_DDI_7 = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=DDI_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_BCN_9 = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=BCN_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )
        
        classifier_SNU_134 = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=SNU_134_CLASSNAMES_ONTOLOGY_AUG if args.ontology_aug else SNU_134_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_SCIN_20 = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=SCIN_20_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )
        
        classifier_SCIN_211 = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=SCIN_211_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )
        
        classifier_SD_128 = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=SD_128_CLASSNAMES_ONTOLOGY_AUG if args.ontology_aug else SD_128_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

        classifier_DAFFODIL_5 = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=DAFFODIL_5_CLASSNAMES_ONTOLOGY_AUG if args.ontology_aug else SD_128_CLASSNAMES,
            templates=templates,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    # dermnet
    if args.zeroshot_eval:
        test_auroc, acc = run(model, classifier_dermnet, data['zeroshot_dermnet'].dataloader, len(DERMNET_CLASSNAMES), args, metric='auroc+acc')
        results['zeroshot-dermnet-auroc'] = test_auroc
        results['zeroshot-dermnet-acc'] = acc

    # pad
    if args.zeroshot_eval1:
        test_auroc, acc = run(model, classifier_pad, data['zeroshot_pad'].dataloader, len(PAD_CLASSNAMES), args, metric='auroc+acc')
        results['zeroshot-pad-auroc'] = test_auroc
        results['zeroshot-pad-acc'] = acc

    # ham
    if args.zeroshot_eval2:
        auroc, acc = run(model, classifier_ham, data['zeroshot_ham'].dataloader, len(HAM_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-ham-auroc'] = auroc
        results['zeroshot-ham-acc'] = acc

    # f17k - 82
    if args.zeroshot_eval3:
        auroc, acc = run(model, classifier_f17k_113_disease, data['zeroshot_f17k_113_disease'].dataloader, len(F17K_DISEASE_113_CLASSES),args, metric='auroc+acc')
        results['zeroshot-f17k-113-auroc'] = auroc
        results['zeroshot-f17k-113-acc'] = acc

    # f17k - 9
    if args.zeroshot_eval4:
        test_auroc, f1 = run(model, classifier_f17k_9_disease, data['zeroshot_f17k_9_disease'].dataloader, len(F17K_DISEASE_9_CLASSES),args)
        results['zeroshot-f17k-9-auroc'] = test_auroc
        results['zeroshot-f17k-9-f1'] = f1

    # skincon 32
    if args.zeroshot_eval5:
        test_auroc, f1 = run(model, classifier_skincon_32_disease, data['zeroshot_skincon_32_disease'].dataloader, len(SKINCON_32_CLASSES),args, True)
        results['zeroshot-skincon-32-auroc'] = test_auroc
        results['zeroshot-skincon-32-f1'] = f1

    # ham - 2
    if args.zeroshot_eval6:
        auroc, acc = run(model, classifier_ham_2, data['zeroshot_ham-2-classes'].dataloader, len(HAM_2_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-ham-2-auroc'] = auroc
        results['zeroshot-ham-2-acc'] = acc
    
    # DDI - 7
    if args.zeroshot_eval7:
        test_auroc, f1 = run(model, classifier_DDI_7, data['zeroshot_DDI-7-classes'].dataloader, len(DDI_CLASSNAMES),args)
        results['zeroshot-DDI-7-auroc'] = test_auroc
        results['zeroshot-DDI-7-f1'] = f1

    # BCN - 9
    if args.zeroshot_eval8:
        test_auroc, f1 = run(model, classifier_BCN_9, data['zeroshot_BCN-9-classes'].dataloader, len(BCN_CLASSNAMES),args)
        results['zeroshot-BCN-9-auroc'] = test_auroc
        results['zeroshot-BCN-9-f1'] = f1

    # SNU - 134
    if args.zeroshot_eval9:
        auroc, acc = run(model, classifier_SNU_134, data['zeroshot_SNU-134-classes'].dataloader, len(SNU_134_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-SNU-134-auroc'] = auroc
        results['zeroshot-SNU-134-acc'] = acc

    # SCIN - 20
    if args.zeroshot_eval10:
        top1_acc, top5_acc = run(model, classifier_SCIN_20, data['zeroshot_SCIN-20-classes'].dataloader, len(SCIN_20_CLASSNAMES),args, metric='acc')
        results['zeroshot-SCIN-20-acc'] = top1_acc
        results['zeroshot-SCIN-20-top5-acc'] = top5_acc

    # SCIN - 211
    if args.zeroshot_eval11:
        top1_acc, top5_acc = run(model, classifier_SCIN_211, data['zeroshot_SCIN-211-classes'].dataloader, len(SCIN_211_CLASSNAMES),args, metric='acc')
        results['zeroshot-SCIN-211-acc'] = top1_acc
        results['zeroshot-SCIN-211-top5-acc'] = top5_acc

    # SD - 128
    if args.zeroshot_eval12:
        auroc, acc = run(model, classifier_SD_128, data['zeroshot_SD-128-classes'].dataloader, len(SD_128_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-SD-128-auroc'] = auroc
        results['zeroshot-SD-128-acc'] = acc
    
    # DAFFODIL - 5
    if args.zeroshot_eval13:
        auroc, acc = run(model, classifier_DAFFODIL_5, data['zeroshot_DAFFODIL-5-classes'].dataloader, len(DAFFODIL_5_CLASSNAMES),args, metric='auroc+acc')
        results['zeroshot-Daffodil-5-auroc'] = auroc
        results['zeroshot-Daffodil-5-acc'] = acc

    logging.info('Finished zero-shot imagenet.')

    return results
