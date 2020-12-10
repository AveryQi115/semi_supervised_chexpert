import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
import ipdb


class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        output = output
        loss = F.cross_entropy(output, target)
        return loss

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, para_dict=None, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        num_classes = para_dict['num_classes']
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)

        self.gamma = gamma
        self.num_classes = num_classes
        self.size_average = size_average

    def forward(self, inputs, targets):
        """
            Args:
                inputs: shape: [N, C]
                targets: shape: [N]
        """
        N, C = inputs.size(0), inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)

        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, para_dict=None):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.para_dict = para_dict
        self.num_classes = para_dict["num_classes"]
        self.epsilon = para_dict.get("epsilon", 0.1)
        self.use_gpu = para_dict['device'] == torch.device('cuda')
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class CSCE(nn.Module):

    def __init__(self, para_dict=None):
        super(CSCE, self).__init__()
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        scheduler = cfg.LOSS.CSCE.SCHEDULER
        self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH

        if scheduler == "drw":
            self.betas = [0, 0.999999]
        elif scheduler == "default":
            self.betas = [0.999999, 0.999999]
        self.weight = None

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def forward(self, x, target, **kwargs):
        return F.cross_entropy(x, target, weight= self.weight)

# this module is for multi-label classification task
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, para_dict=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.cfg = para_dict['cfg']
        self.num_classes = para_dict['num_classes']
        self.num_class_list = para_dict['num_class_list']
        self.device = para_dict['device']

    # TODO: change the evaluation step of ACC
    def forward(self, output, target):
        assert output.shape[1] == self.num_classes, f'output.shape:{output.shape}'
        assert output.dtype == torch.float32 or output.dtype==torch.float16,f'output.dtype={output.dtype}'
        pos_weight = torch.from_numpy(
            np.array(self.num_class_list,
                     dtype=np.float32)).to(self.device).type_as(output)
        target = target.to(self.device).type_as(output)
        # ipdb.set_trace()
        loss_list = []
        for i in range(self.num_classes):
            target_i = target[:, i].view(-1)
            loss_list.append(F.binary_cross_entropy_with_logits(output[:,i].view(-1), target_i,pos_weight[i]))
        loss = torch.sum(torch.stack(loss_list))
        return loss

# TODO: create BCEWithLogitsLoss with mask for U-ignore


# The LDAMLoss class is copied from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).
class LDAMLoss(nn.Module):

    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        max_m = cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        assert s > 0

        self.s = s
        self.step_epoch = cfg.LOSS.LDAM.DRW_EPOCH
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight= self.weight)