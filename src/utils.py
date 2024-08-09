import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test


def cross_entropy_loss(device, logits, grade, **kwargs):
    grade = grade.float().to(device)

    return nn.CrossEntropyLoss()(logits, grade)


# train_accuracy = accuracy_score(train_y, train_pred.argmax(axis=1))

# if self.num_classes == 2:
#     train_auc_score = roc_auc_score(train_y, train_pred)
# else:
#     train_auc_score = roc_auc_score(train_y, train_pred, multi_class='ovr')


def get_accuracy(grade, Y_prob, **kwargs):
    grade = grade.argmax(axis=1).cpu().detach().numpy().squeeze()
    Y_prob = Y_prob.argmax(axis=1).cpu().detach().numpy().squeeze()

    return accuracy_score(grade, Y_prob)


def get_roc_auc_score(grade, Y_prob, num_classes, **kwargs):
    grade = grade.cpu().detach().numpy().squeeze()
    Y_prob = Y_prob.cpu().detach().numpy().squeeze()
    num_classes = num_classes.cpu().detach().numpy()[0]
    # num_classes = np.unique(grade.argmax(axis=1)).shape[0]

    # print("Number of Classes", num_classes)
    # print(grade)
    # print(Y_prob)
    try:
        if num_classes == 2:
            rocauc_score = roc_auc_score(grade, Y_prob)
        else:
            rocauc_score = roc_auc_score(grade, Y_prob, multi_class='ovr')
        return rocauc_score
    except Exception as e:
        # print(e)
        # print(num_classes)
        # print(grade.argmax(axis=1), Y_prob.argmax(axis=1))
        return None

    # return rocauc_score


def get_metric_fn(metric_name):
    if metric_name == "cindex":
        metric_fn = CIndex_lifeline
    elif metric_name == "pvalue":
        metric_fn = cox_log_rank
    elif metric_name == "survival_accuracy":
        metric_fn = accuracy_cox
    elif metric_name == "accuracy":
        metric_fn = get_accuracy
    elif metric_name == "roc_auc_score":
        metric_fn = get_roc_auc_score
    else:
        raise NotImplementedError(
            f"The evaluation metric {metric_name} is not implemented")

    return metric_fn


def get_loss_fn(loss_name):
    if loss_name == "cross_entropy":
        loss_fn = cross_entropy_loss
    elif loss_name == "cox":
        loss_fn = cox_loss
    else:
        raise NotImplementedError(
            f"The loss function {loss_name} is not implemented")

    return loss_fn


def get_optimizer_fn(optimizer_name, lr, parameters):
    if optimizer_name == "adam":
        optimizer = optim.Adam(parameters, lr=lr)
    elif optimizer_name == "radam":
        optimizer = optim.RAdam(parameters, lr=lr)
    else:
        raise NotImplementedError(
            f"The optimizer function {optimizer_name} is not implemented")

    return optimizer


def calculate_loss(task, losses, device, **kwargs):
    total_loss = 0
    for loss in losses:
        losses[loss]["value"] = get_loss_fn(loss)(device, **kwargs)
        total_loss = total_loss + losses[loss]["value"]
    # print("KWARGS",kwargs)
    # total_loss = 0
    # for loss_function in loss_functions:
    #     total_loss = total_loss + loss_function(device, **kwargs)

    return total_loss


def cox_loss(device, survtime, censor, hazard, **kwargs):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard.reshape(-1).to(device)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) *
        censor.to(device))
    return loss_cox


def CIndex_lifeline(hazard, censor, survtime, **kwargs):
    if torch.is_tensor(hazard):
        hazard = hazard.cpu().detach().numpy()
    if torch.is_tensor(censor):
        censor = censor.cpu().detach().numpy()
    if torch.is_tensor(survtime):
        survtime = survtime.cpu().detach().numpy()
    try:
        return (concordance_index(survtime, -hazard, censor))
    except:
        None


def cox_log_rank(hazard, censor, survtime, **kwargs):
    if torch.is_tensor(hazard):
        hazard = hazard.cpu().detach().numpy().reshape(-1)
    if torch.is_tensor(censor):
        censor = censor.cpu().detach().numpy().reshape(-1)
    if torch.is_tensor(survtime):
        survtime = survtime.cpu().detach().numpy().reshape(-1)
    # print("hazard",hazard.shape)
    # print("censor",censor.shape)
    # print("survtime",survtime.shape)
    # yada

    median = np.median(hazard)
    hazards_dichotomize = np.zeros([len(hazard)], dtype=int)
    hazards_dichotomize[hazard > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime[idx]
    T2 = survtime[~idx]
    E1 = censor[idx]
    E2 = censor[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)


def accuracy_cox(hazard, censor, **kwargs):
    if torch.is_tensor(hazard):
        hazard = hazard.cpu().detach().numpy().reshape(-1)
    if torch.is_tensor(censor):
        censor = censor.cpu().detach().numpy().reshape(-1)

    # print(hazard.shape, censor.shape)

    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazard)
    hazards_dichotomize = np.zeros([len(hazard)], dtype=int)
    hazards_dichotomize[hazard > median] = 1
    correct = np.sum(hazards_dichotomize == censor)
    return correct / len(censor)


def calculate_metrics(task, evaluators, epoch_logs, **kwargs):

    print("Epoch Logs", epoch_logs[0].keys())
    # Epoch Logs dict_keys(['grade', 'censor', 'survtime', 'hazard', 'logits', 'Y_hat', 'Y_prob'])

    # for x in epoch_logs:
    #     print(x)
    #     break
    # print("Logs Len",len(epoch_logs))
    # yada
    # for evaluator in evaluators:
