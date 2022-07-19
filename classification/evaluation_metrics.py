import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics import AUROC
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, \
    balanced_accuracy_score, precision_score, recall_score, f1_score, fbeta_score


def predictions(model, target_loader, device):
    """
    Calculates predictions on test dataset
    returns list of probabilities, true and predicted labels
    """
    model.to(device)

    true_labels = []
    pred_labels = []
    probs = []
    tensor_probs = []

    for i, (images, labels, path) in tqdm(enumerate(target_loader),
                                          total=len(target_loader)):
        #     for i, (images, labels, path) in enumerate(tqdm(target_loader)):
        images = images.to(device)
        labels = labels.to(device)

        true_labels = true_labels + labels.tolist()
        with torch.no_grad():
            model.eval()
            out = model(images)

            _, pred = torch.max(out, 1)
            preds = np.squeeze(pred.clone().detach())
            
            #  get probabilities tensor for AUROC computing
            class_probs_batch = [F.softmax(el, dim=0) for el in out]

            # get probabilities list
            batch_probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, out)]
            
            # group tensor probs
            tensor_probs.append(class_probs_batch)

            # group together probabilities in batch
            probs = probs + batch_probs

            # group together predictions in batch
            pred = pred.tolist()

            pred_labels = pred_labels + pred

    return true_labels, pred_labels, probs, tensor_probs


class Metrics(object):
    """
    Computes classification metrics for the test subset: Acc, BA, F1, F2,
    MCC, confusion matrix, classification report sensitivity, specificity,
    precision and auroc.
    """

    def __init__(self, y_true, y_pred, tensor_prob, classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.prob = tensor_prob
        self.classes = sorted(classes)
        self.cf_matrix = confusion_matrix(self.y_true, self.y_pred)

    def mcc(self):
        return round(matthews_corrcoef(self.y_true, self.y_pred), 4)

    def balanced_acc(self):
        return round(balanced_accuracy_score(self.y_true, self.y_pred), 4)

    def classification_rep(self):
        target_names = sorted(self.classes)
        return classification_report(self.y_true, self.y_pred, target_names=target_names, digits=3)

    def cmatrix(self):
        df_cm = pd.DataFrame(self.cf_matrix, index=sorted(self.classes),
                             columns=sorted(self.classes))
        return self.show_confusion_matrix(df_cm)

    def micro_average(self):
        micro_pres = precision_score(self.y_true, self.y_pred, average='micro')
        micro_rec = recall_score(self.y_true, self.y_pred, average='micro')
        micro_f1 = f1_score(self.y_true, self.y_pred, average='micro')
        return round(micro_pres, 4), round(micro_rec, 4), round(micro_f1, 4)

    def macro_average(self):
        macro_pres = np.round(precision_score(self.y_true, self.y_pred, average='macro'), 4)
        macro_rec = np.round(recall_score(self.y_true, self.y_pred, average='macro'), 4)
        macro_f1 = np.round(f1_score(self.y_true, self.y_pred, average='macro'), 4)
        macro_f2 = np.round(fbeta_score(self.y_true, self.y_pred, average='macro', beta=2), 4)
        f1_class = np.round(f1_score(self.y_true, self.y_pred, average=None), 4)
        f2_class = np.round(fbeta_score(self.y_true, self.y_pred, average=None, beta=2), 4)
        return macro_pres, macro_rec, macro_f1, macro_f2, f1_class, f2_class

    @staticmethod
    def show_confusion_matrix(cm):
        hmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        # plt.figure(figsize=(10,4))
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

    def sensitivity_specificity(self):
        fp = self.cf_matrix.sum(axis=0) - np.diag(self.cf_matrix)
        fn = self.cf_matrix.sum(axis=1) - np.diag(self.cf_matrix)
        tp = np.diag(self.cf_matrix)
        tn = self.cf_matrix.sum() - (fp + fn + tp)
        # sensitivity, recall, or true positive
        sensitivity = np.round(tp / (tp + fn), 4)
        # specificity or true negative rate
        specificity = np.round(tn / (tn + fp), 4)
        return sensitivity, specificity, (tp, tn, fp, fn)
        
    def auroc(self):
        probs = torch.cat([torch.stack(batch) for batch in self.prob])
        prob = probs.clone().detach()
        true = torch.as_tensor(self.y_true)
        auroc = AUROC(num_classes=len(self.classes))
        aur_ = auroc(prob.to('cpu'), true.to('cpu'))
        return round(float(aur_), 4)


def misclassification(y_true, y_pred, probs, df_test):
    """
    :param y_true: list of true labels
    :param y_pred:  list of predictions
    :param probs: list of probabilities
    :param df_test:  path to test labels
    :return: a dataframe
    """
    test_path = list(df_test['image_path'])

    miss_class = {}
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            miss_class[test_path[i]] = {'image_path': test_path[i], 'true_label': y_true[i],
                                        'predicted_label': y_pred[i], 'probability': probs[i]}

    df_misses = pd.DataFrame(miss_class.values())

    return df_misses


def train_metrics(train_dict):
    """
    :param train_dict: a dictionary with the accuracy and loss logs for training and validation
    :return:  max, min accuraciies and the number of epoch to reach the maximum validation accuracy
    """
    tr_acc = train_dict['train_acc'][-1]
    val_acc = train_dict['val_acc'][-1]
    max_acc = max(train_dict['val_acc'])
    max_index = train_dict['val_acc'].index(max_acc)
    val_loss = train_dict['val_loss'][-1]
    min_loss = min(train_dict['val_loss'])
    loss_idx = train_dict['val_loss'].index(min_loss)
    return tr_acc, val_acc, max_acc, max_index, val_loss, min_loss, loss_idx