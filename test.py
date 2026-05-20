import torch

from metric import get_metrics


@torch.no_grad()
def model_test(model, test_edge_index, test_label, output):
    model.eval()  # 设置为评估模式

    link_logits = model.decode(output, test_edge_index)  # 得到预测值,是一个行向量
    link_probs = torch.nn.functional.sigmoid(link_logits)  # 把预测值转化为数值在0到1之间的概率值
    # 通过使用get_metrics函数来计算评价指标,参数为真实标签和预测概率
    auc, acc, aupr, precision, recall, f1, mcc, tpr, precision1 = get_metrics(test_label, link_probs)

    return auc, acc, aupr, precision, recall, f1, mcc, tpr, precision1





