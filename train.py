import torch

# 传入的参数依次为模型，优化器，损失函数，数据，训练集边，训练集标签
def model_train(model, optimizer, loss_function, data, drs_data, dis_data, train_edge_index, train_label, lr_scheduler):
    model.train()
    optimizer.zero_grad()   # 梯度清零

    # 添加 detach() 防止计算图保留
    output = model.forward(data, drs_data, dis_data)  # 若是C数据集，则output大小为663*409
    link_logits = model.decode(output, train_edge_index)  # 一个一维张量

    loss = loss_function(link_logits, train_label)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    return loss, output






