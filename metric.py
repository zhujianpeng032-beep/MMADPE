import numpy as np
from sklearn import metrics

# real_score:真实标签， predict_score:预测分数
def get_metrics(real_score, predict_score):


    # 假设 pre_scores 是一个一维的 NumPy 数组
    pre_scores = np.squeeze(predict_score.cpu().detach().numpy())  # 预测分数

    real_labels = np.squeeze(real_score.cpu().detach().numpy())     # 真实标签

    fpr, tpr, thresholds = metrics.roc_curve(real_labels, pre_scores)
    precision1, recall1, _ = metrics.precision_recall_curve(real_labels, pre_scores)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.auc(recall1, precision1)

    # 使用一个列表推导式，根据pre_scores中的每个元素j的值，如果j小于0.5，则将0添加到pred_labels列表中，否则将1添加到列表中。简单来说，它根据阈值0.5将预测分数转化为预测标签。
    pred_labels = [0 if j < 0.5 else 1 for j in pre_scores]  # 预测标签(根据预测分数转化而成)
    acc = metrics.accuracy_score(real_labels, pred_labels)
    precision = metrics.precision_score(real_labels, pred_labels)
    recall = metrics.recall_score(real_labels, pred_labels)
    f1 = metrics.f1_score(real_labels, pred_labels)
    mcc = metrics.matthews_corrcoef(real_labels, pred_labels)

    return auc, acc, aupr, precision, recall, f1, mcc, tpr, precision1


