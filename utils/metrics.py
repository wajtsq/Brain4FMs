# coding: UTF-8
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, top_k_accuracy_score, cohen_kappa_score
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


class BinaryClassMetrics:
    def __init__(self, args, pred, prob, true, ):
        y_score = softmax(prob)
        y_true = np.asarray(true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float32)
        y_pred = np.argmax(prob, axis=-1)
        
        if not isinstance(pred, np.ndarray):
            pred, prob, true = pred.detach().numpy(), prob.detach().numpy(), true.detach().numpy()

        if args.dataset == 'Clinical' and args.run_mode == 'test':
            # remove label==2 in the Clinical inferring file
            pred = pred[true != 2]
            prob = prob[true != 2]
            true = true[true != 2]

        if np.sum(true) == 0 and np.sum(pred) == 0:
            self.special_good = True
        else:
            self.special_good = False

        self.tn = np.sum((pred == 0) & (true == 0))
        self.tp = np.sum((pred == 1) & (true == 1))
        self.fn = np.sum((pred == 0) & (true == 1))
        self.fp = np.sum((pred == 1) & (true == 0))
        # self.acc = np.sum(pred == true) / len(true)  # (tp+tn) / (tp+tn+fp+fn)
        # self.prec = self.tp / (self.tp + self.fp)
        # self.rec = self.tp / (self.tp + self.fn)
        # self.f_half = self.fbeta(self.prec, self.rec, beta=0.5)
        # self.f_one = self.fbeta(self.prec, self.rec, beta=1)
        # self.f_doub = self.fbeta(self.prec, self.rec, beta=2)
        self.acc   = accuracy_score(y_true, y_pred)
        self.bacc  = balanced_accuracy_score(y_true, y_pred)
        self.prec  = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
        self.rec   = recall_score(y_true, y_pred,    pos_label=1, average='binary', zero_division=0)
        self.f_one = fbeta_score(y_true, y_pred, beta=1, pos_label=1, average='binary', zero_division=0)
        self.f_doub= fbeta_score(y_true, y_pred, beta=2, pos_label=1, average='binary', zero_division=0)
        self.f_half= fbeta_score(y_true, y_pred, beta=0.5, pos_label=1, average='binary', zero_division=0)
        self.f_one_macro  = fbeta_score(y_true, y_pred, beta=1, labels=np.arange(2), average='macro')
        self.f_doub_macro = fbeta_score(y_true, y_pred, beta=2, labels=np.arange(2), average='macro')

        precision, recall, thresholds = precision_recall_curve(y_true, y_score[:, 1])
        self.auprc = auc(recall, precision)

        self.auroc = roc_auc_score(y_true, y_score[:,1])
        self.bacc = balanced_accuracy_score(y_true, y_pred)

        self.conf_matrix = self.get_confusion()

    def get_confusion(self):
        return f"TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn} " if not self.special_good else "special_good"

    # def get_metrics(self, one_line=False):
    #     if one_line:
    #         out = 'Acc:%.4f Prec:%.4f Rec:%.4f F1:%.4f F2:%.4f AUPRC:%.4f AUROC:%.4f' \
    #               % (self.acc, self.prec, self.rec, self.f_one, self.f_doub, self.auprc, self.auroc)
    #     else:
    #         out = ''
    #         out += '-' * 15 + 'Metrics' + '-' * 15 + '\n'
    #         out += 'Accuracy:  ' + str(self.acc) + '\n'
    #         out += 'Precision: ' + str(self.prec) + '\n'
    #         out += 'Recall:    ' + str(self.rec) + '\n'
    #         out += 'F1:        ' + str(self.f_one) + '\n'
    #         out += 'F2:        ' + str(self.f_doub) + '\n'
    #         out += 'AUROC:     ' + str(self.auprc) + '\n'
    #         out += 'AUPRC:     ' + str(self.auroc) + '\n'
    #     return out if not self.special_good else "special_good"
    #
    # @staticmethod
    # def fbeta(precision, recall, beta):
    #     return (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)

def compute_class_metrics(true, pred, l, basic_args=None):
        res = []
        pred = np.argmax(pred, axis=1)
        binary_true = (true == l)
        binary_pred = (pred == l)

        cm = confusion_matrix(binary_true, binary_pred, labels=[False, True])
        if cm.shape != (2, 2): 
            tn, fp, fn, tp = 0, 0, 0, 0
            if binary_true.sum() == 0:  # neg
                tn = cm[0][0]
                fp = cm[0][1] if cm.shape[1] > 1 else 0
            elif binary_true.sum() == len(binary_true):  # pos
                tp = cm[0][0]
                fn = cm[0][1] if cm.shape[1] > 1 else 0
        else:
            tn, fp, fn, tp = cm.ravel()

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        res.append([spec, sens])
        return res
        


class MultiClassMetrics:
    def __init__(self, basic_args, pred, prob, true):
        num_class = basic_args.n_class
        pred, prob, true = pred.detach().numpy(), prob.detach().numpy(), true.detach().numpy()
        y_true = np.asarray(true, dtype=np.int64)
        y_score = softmax(np.asarray(prob, dtype=np.float32), axis=1)
        y_pred = np.argmax(prob, axis=-1)
        
        k = getattr(basic_args, 'k', min(3, num_class-1))

        self.k = k
        self.acc = top_k_accuracy_score(y_true, y_score, k=k, labels=np.arange(num_class))
        if len(np.unique(y_true)) != num_class:
            self.auc_roc_macro = self.hand_till_multiclass_auc(y_true, y_score, num_class)
        else:
            self.auc_roc_macro = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr', labels=np.arange(num_class))
        self.prec = precision_score(y_true, y_pred, labels=np.arange(num_class), average='macro')
        self.rec = recall_score(y_true, y_pred, labels=np.arange(num_class), average='macro')
        self.f_one_macro  = fbeta_score(y_true, y_pred, beta=1, labels=np.arange(num_class), average='macro')
        self.f_doub_macro = fbeta_score(y_true, y_pred, beta=2, labels=np.arange(num_class), average='macro')
        res = []
        for l in range(num_class):
            prec, recall, _, _ = precision_recall_fscore_support(true == l,
                                                                 pred == l,
                                                                 pos_label=True, average=None, zero_division=0)
            if len(recall) < 2:
                recall = compute_class_metrics(true, prob, l)[0]
            res.append([recall[0], recall[1]])  # 0:spec 1:sens
        res = np.array(res)
        self.spec_mean = np.mean(res[:, 0])
        self.sens_mean = np.mean(res[:, 1])

        self.kappa = cohen_kappa_score(y_true, y_pred, labels=np.arange(num_class))

        self.conf_matrix = ('\n' + str(confusion_matrix(y_true, y_pred))) if hasattr(basic_args, 'print_matrix') and basic_args.print_matrix else ''
        self.accuracy = accuracy_score(y_true, y_pred)
        self.f1_scores = f1_score(y_true, y_pred, average=None)
        self.bacc = balanced_accuracy_score(y_true, y_pred)

    def get_confusion(self):
        return self.conf_matrix
    
    def _ensure_probabilities(self, prob):
        prob_sums = prob.sum(axis=1)
        if not np.allclose(prob_sums, 1.0, atol=1e-3):
            prob_normalized = torch.softmax(torch.tensor(prob), dim=1).numpy()
            return prob_normalized
        return prob

    def get_metrics(self, one_line=False):
        if one_line:
            out = f'Top {self.k} Acc:%.4f' % self.acc  + '\n'
            out += 'Sens:%.4f Spec:%.4f macroF1:%.4f Kappa:%.4f' % (self.sens_mean, self.spec_mean, self.f_one_macro, self.kappa) + '\n'
            out += 'F1 of each label: ' + np.array2string(self.f1_scores, precision=4,)
        else:
            out = ''
            out += '-' * 8 + f'Multiclass Metrics (Top {self.k} Acc)' + '-' * 8 + '\n'
            out += 'Accuracy:  ' + str(self.acc) + '\n'
            out += 'Sensitivity: ' + str(self.sens_mean) + '\n'
            out += 'Specificity: ' + str(self.spec_mean) + '\n'
            out += 'macro F1     ' + str(self.f_one_macro) + '\n'
            out += 'Kappa:       ' + str(self.kappa) + '\n'
        return out

    def draw_conf_matrix(self, config, annot=False, epoch=None):
        import seaborn as sb
        import pandas as pd
        # class_name = np.unique(np.concatenate([np.unique(self.pred), np.unique(self.true)]))
        class_name = np.unique(np.unique(self.true))

        cm_df = pd.DataFrame(self.conf_matrix,
                             index=class_name,
                             columns=class_name)

        f, (ax1, axcb) = plt.subplots(1, 2, figsize=(18, 18), gridspec_kw={'width_ratios': [1, 0.04]})
        cm_df = cm_df.apply(lambda x: (x / x.sum()), axis=1)
        g1 = sb.heatmap(cm_df, ax=ax1, cbar_ax=axcb, cmap="Blues", annot=annot, fmt='.2f', square=True)

        # g1 = sb.heatmap(cm_df, ax=ax1, cbar_ax=axcb, cmap="Blues", annot=annot, fmt='d', square=True)

        g1.set_title(f'exp{config.exp_id}_epoch{epoch}' if epoch is not None else \
                     f'exp{config.exp_id}')
        g1.set_ylabel('label')
        g1.set_xlabel('pred')
        plt.show()
        
    def hand_till_multiclass_auc(self, y_true, y_score, classes):
        import itertools
        y_true = np.asarray(y_true)
        present = np.intersect1d(range(classes), np.unique(y_true))
        if len(present) < 2:
            return np.nan

        aucs = []
        for i, j in itertools.combinations(present, 2):
            mask = np.isin(y_true, [i, j])
            yt = y_true[mask]
            ps = y_score[mask]
            # i vs j
            y_bin_i = (yt == i).astype(int)
            y_bin_j = (yt == j).astype(int)
            # p_i, p_j
            auc_i = roc_auc_score(y_bin_i, ps[:, i])
            auc_j = roc_auc_score(y_bin_j, ps[:, j])
            aucs.append(0.5 * (auc_i + auc_j))
        return float(np.mean(aucs))

