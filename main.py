import argparse
import time
import pandas as pd
from collections import defaultdict
from test import model_test
from train import model_train
from utils import *
from model import Model
from self_define_loss import *
from scipy import interpolate
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    # parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout')

    parser.add_argument('--disease_index', default=136, type=int, help='disease index')  # 专门用于做案例研究

    parser.add_argument('--hops', type=int, default=5, help='Hop of neighbors to be calculated')
    parser.add_argument('--similarity', type=int, default=0.3, help='limitiation of similarity')

    parser.add_argument('--drug_GMA_input_dim', default=128, type=int, help='input_channels(GraphMamba)')  # 要跟的低维嵌入的输出维数保持一致
    parser.add_argument('--drug_GMA_output_dim', default=64, type=int, help='output_channels(GraphMamba)')

    parser.add_argument('--dis_GMA_input_dim', default=128, type=int, help='input_channels(GraphMamba)')  # 要跟低维嵌入的输出维数保持一致
    parser.add_argument('--dis_GMA_output_dim', default=64, type=int, help='output_channels(GraphMamba)')

    parser.add_argument('--GGPS_input_dim', default=64, type=int, help='input_channels(GraphGPS)')
    parser.add_argument('--GGPS_output_dim', default=64, type=int, help='output_channels(GraphGPS)')

    parser.add_argument('--feature_output', type=int, default=64, help='output dimensions of node features.')
    parser.add_argument('--tot_updates', type=int, default=1000, help='used for optimizer learning rate scheduling')  # 学习率调度器
    parser.add_argument('--warmup_updates', type=int, default=400, help='warmup steps')

    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')

    parser.add_argument('--peak_lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.000002, help='Final learning rate')

    args = parser.parse_args()  # 获取所有参数
    args.data_dir = '../data/' + args.dataset + '/'
    data = get_data(args)  # 获取数据

    args.drug_number = data['drug_number']  # 获取药物数量
    args.disease_number = data['disease_number']  # 获取疾病数量

    data, drs_data, dis_data = data_processing(data, args)  # 数据预处理
    data = k_fold(data, args)  # 获取所有的训练集,测试集,以及他们对应的标签
    # data = case_study(data, args)

    data = move_to_device(data, Device)  # 把data中所有数据转移到指定的设备Device上
    drs_data = drs_data.to(Device)
    dis_data = dis_data.to(Device)

    loss_function = weighted_cross_entropy_loss

    header = '{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(
        'Epoch', 'Time', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Mcc', 'loss'
    )

    AUCs, AUPRs, ACCs, Precisions, Recalls, F1s, Mccs, Tprs, Precision1s = [], [], [], [], [], [], [], [], []
    # 开始计时
    start_time = time.time()
    print('Dataset:{}'.format(args.dataset))
    for i in range(args.k_fold):

        print('fold:', i)
        print(header)

        model = Model(args)  # 实例化模型
        model = model.to(Device)  # 将模型移动到GPU上

        # 创建一个优化器,参数为模型参数和学习率以及权重衰减
        optimizer = torch.optim.Adam(model.parameters(), lr=args.peak_lr)

        lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0
        )

        best_auc = 0
        best_aupr = 0
        # 获取第i折的训练集和测试集以及它们对应的标签
        X_train = torch.LongTensor(data['X_train'][i]).to(Device)  # 训练集索引
        Y_train = torch.LongTensor(data['Y_train'][i]).to(Device).flatten()  # 训练集标签
        X_test = torch.LongTensor(data['X_test'][i]).to(Device)  # 测试集索引
        Y_test = torch.LongTensor(data['Y_test'][i]).to(Device).flatten()  # 测试集标签

        for epoch in range(args.epochs):
            # output是模型前馈网络最后生成的663*409的矩阵，它是一个分数矩阵
            loss, output = model_train(model, optimizer, loss_function, data, drs_data, dis_data, X_train.T, Y_train, lr_scheduler)


            auc, acc, aupr, precision, recall, f1, mcc, tpr, precision1= model_test(model, X_test.T, Y_test, output)


            # 计算并打印总共训练所花费的时间
            time1 = time.time() - start_time

            metrics = [epoch, time1, auc, aupr, acc, precision, recall, f1, mcc, loss]
            row = '{:<10}{:<10.2f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.5f}{:<10.4f}{:<10.5f}{:<10.4f}{:<10.5f}'.format(*metrics)
            print(row)

            if auc > best_auc:
                best_epoch = epoch + 1
                best_auc = auc
                # 当AUC值是最好的时候，默认AUPR值和其他指标也是最好的
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc, best_tpr, best_precision1 = aupr, acc, precision, recall, f1, mcc, tpr, precision1
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc, ';\tbest_aupr:', best_aupr)
        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        ACCs.append(best_accuracy)
        Precisions.append(best_precision)
        Recalls.append(best_recall)
        F1s.append(best_f1)
        Mccs.append(best_mcc)
        Tprs.append(best_tpr)
        Precision1s.append(best_precision1)


    # 插值对齐 TPR 长度
    if Tprs:
        target_length = 2000
        interpolated_tprs = []
        for tpr in Tprs:
            original_length = len(tpr)
            if original_length != target_length:
                # 使用线性插值对齐长度
                original_thresholds = np.linspace(0, 1, original_length)
                interp_func = interpolate.interp1d(original_thresholds, tpr, kind='linear',
                                                   fill_value='extrapolate')
                interpolated_tpr = interp_func(np.linspace(0, 1, target_length))
                interpolated_tprs.append(interpolated_tpr)
            else:
                interpolated_tprs.append(tpr)

        avg_tpr = np.mean(interpolated_tprs, axis=0)
        print('Average TPR:', avg_tpr.shape)

        # 创建一个 DataFrame 用于保存 avg_tpr
        tpr_df = pd.DataFrame({'Change2': avg_tpr})

        # 将 DataFrame 保存到 Excel 文件，使用 openpyxl 引擎
        tpr_df.to_excel('tpr_results.xlsx', index=False, engine='openpyxl')
        print("TPR results saved to 'tpr_results.xlsx'")




    if Precision1s:
        target_length = 2000
        interpolated_Precision1s = []

        for pre in Precision1s:
            original_length = len(pre)

            # 确保Precision是降序排列（对应Recall升序）
            pre_sorted = np.array(pre)[::-1]  # 反转数组

            if original_length != target_length:
                # 使用线性插值对齐长度（从大到小对应Recall 0→1）
                original_thresholds = np.linspace(0, 1, original_length)
                interp_func = interpolate.interp1d(
                    original_thresholds, pre_sorted,
                    kind='linear', fill_value='extrapolate'
                )
                interpolated_Precision1 = interp_func(np.linspace(0, 1, target_length))
                interpolated_Precision1s.append(interpolated_Precision1)
            else:
                interpolated_Precision1s.append(pre_sorted)  # 直接使用反转后的数据

        # 计算平均Precision（保持降序）
        avg_pre = np.mean(interpolated_Precision1s, axis=0)
        print('Average Precision shape:', avg_pre.shape)  # 应为 (2000,)

        # 保存到Excel（列名可自定义）
        pre_df = pd.DataFrame({'Precision': avg_pre})
        pre_df.to_excel('pre_results.xlsx', index=False, engine='openpyxl')
        print("Precision results saved to 'pre_results.xlsx'")

    print_metrics('AUC', AUCs)
    print_metrics('AUPR', AUPRs)
    print_metrics('ACC', ACCs)
    print_metrics('Precision', Precisions)
    print_metrics('Recall', Recalls)
    print_metrics('F1', F1s)
    print_metrics('Mcc', Mccs)




















