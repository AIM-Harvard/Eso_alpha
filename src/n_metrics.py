from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score
import wandb
import torch

def compute_metrics2(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    f1_class = precision_recall_fscore_support(labels, preds)[2]
    acc = accuracy_score(labels, preds)
    zero, one= f1_class[0], f1_class[1]
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    roc_auc = roc_auc_score(labels, preds, average='weighted')
    auprc = average_precision_score(labels, preds, average='weighted')
    print('### class f1 ###')
    print(f1_class)
    print(list(labels))
    print(list(preds))
    wandb.log({
                # 'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                'macro_f1': macro_f1,
                'bclass_0': zero,
                'bclass_1': one,
                'roc_auc': roc_auc,
                'auprc': auprc 
                }) 
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1,
        'bclass_0': zero,
        'bclass_1': one,
        'roc_auc': roc_auc,
        'auprc': auprc 
    }

def compute_metrics22(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    f1_class = precision_recall_fscore_support(labels, preds)[2]
    acc = accuracy_score(labels, preds)
    zero, one= f1_class[0], f1_class[1]
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    roc_auc = roc_auc_score(labels, preds, average='weighted')
    auprc = average_precision_score(labels, preds, average='weighted')
    print('### class f1 ###')
    print(f1_class)
    print(list(labels))
    print(list(preds))
    print({     'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                'macro_f1': macro_f1,
                'bclass_01': zero,
                'bclass_23': one,
                'roc_auc': roc_auc,
                'auprc': auprc 
                })
    wandb.log({
                'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                'macro_f1': macro_f1,
                'bclass_01': zero,
                'bclass_23': one,
                'roc_auc': roc_auc,
                'auprc': auprc 
                }) 
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1,
        'bclass_01': zero,
        'bclass_23': one,
        'roc_auc': roc_auc,
        'auprc': auprc 
    }

def compute_metrics222(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    f1_class = precision_recall_fscore_support(labels, preds)[2]
    acc = accuracy_score(labels, preds)
    zero, one= f1_class[0], f1_class[1]
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    roc_auc = roc_auc_score(labels, preds, average='weighted')
    auprc = average_precision_score(labels, preds, average='weighted')
    print('### class f1 ###')
    print(f1_class)
    print({     'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                'macro_f1': macro_f1,
                'bbclass_1': zero,
                'bbclass_23': one,
                'roc_auc': roc_auc,
                'auprc': auprc 
                })
    wandb.log({
                'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                'macro_f1': macro_f1,
                'bbclass_1': zero,
                'bbclass_23': one,                
                'roc_auc': roc_auc,
                'auprc': auprc 
                }) 
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1,
        'bbclass_01': zero,
        'bbclass_23': one,
        'roc_auc': roc_auc,
        'auprc': auprc 
    }

def compute_metrics3(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    f1_class = precision_recall_fscore_support(labels, preds)[2]
    acc = accuracy_score(labels, preds)
    one, two, three = f1_class[0], f1_class[1], f1_class[2]
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    print('### class f1 ###')
    print(f1_class)
    print(list(labels))
    print(list(preds))    
    print({     'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                'macro_f1': macro_f1,
                'class_0': one,
                'class_1': two,
                'class_23': three
                })
    wandb.log({
                'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                'macro_f1': macro_f1,
                # 'class_0': zero,
                'class_0': one,
                'class_1': two,
                'class_23': three
                }) 

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1,
        'class_0': one,
        'class_1': two,
        'class_23': three
    }

def compute_metrics4(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    f1_class = precision_recall_fscore_support(labels, preds)[2]
    acc = accuracy_score(labels, preds)
    zero, one, two, three = f1_class[0], f1_class[1], f1_class[2], f1_class[3]
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    print('### class f1 ###')
    print(f1_class)
    wandb.log({
                'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                'macro_f1': macro_f1,
                'class_0': zero,
                'class_1': one,
                'class_2': two,
                'class_3': three
                }) 
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }

def compute_metrics6(pred):
    # for multi-label one hot labels setting
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions))
    preds = torch.where(preds>0.5, 1., 0.).int().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='samples')
    f1_class = precision_recall_fscore_support(labels, preds)[2]
    b_zero, b_one, zero, one, two, three, = f1_class[0], f1_class[1], f1_class[2], f1_class[3], f1_class[4], f1_class[5]
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    # one, two, three = f1_class[0], f1_class[1], f1_class[2]
    acc = accuracy_score(labels, preds)
    wandb.log({
                # 'micro_f1': precision_recall_fscore_support(labels, preds, average='micro')[2],
                # 'macro_f1': precision_recall_fscore_support(labels, preds, average='macro')[2],
                'bclass_0': b_zero,
                'bclass_1': b_one,
                'class_0': zero,
                'class_1': one,
                'class_2': two,
                'class_3': three
                }) 
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }
