import numpy as np
import torch, math
from core.evaluate import accuracy
import torch.nn.functional as F

def coteaching_accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # print(f'logit.shape:{logit.shape}')
    output = F.softmax(logit, dim=1)
    # print(f'output.shape:{output.shape}')
    # output.shape=(N,K) K means K class

    maxk = max(topk)
    # print(f'maxk:{maxk}')
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # print(f'pred.shape{pred.shape}')
    # pred.shape = (N, K') K' means top-5 class
    pred = pred.t()
    # print(f'pred.shape{pred.shape}')
    # pred.shape = (K',N) K' means top-5 class(top-5 prediction for item)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct.shape = (K',N) each entry means if the top-K' answer is right for the item
    # print(f'correct.shape:{correct}')

    res = []
    for k in topk:
        # correct.shape=(maxk,N)
        # correct_k.shape = (k,N)
        correct_k = correct[:k]
        correct_k = correct_k.reshape(-1)
        correct_k = correct_k.float().sum(0, keepdim=True)
        # correct_k is the sum of k*batch_size numbers of answer
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def loss_coteaching(y_1, y_2, t, criterion, forget_rate, ind = None, noise_or_not=None):

    # sort loss_1 according to the order of loss,ascending
    loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
    ind_1_sorted = torch.argsort(loss_1).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
    ind_2_sorted = torch.argsort(loss_2).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    # remember_rate will increase as epoch grows
    remember_rate = 1 - forget_rate
    # the longer you train, the less losses model will use to update parameters
    num_remember = int(remember_rate * len(loss_1_sorted))

    # record the right labels rate among all the remembered labels
    if noise_or_not != None and ind != None:
        pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember].cpu()]])/float(num_remember)
        pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember].cpu()]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    # use labels picked by another model to update weights
    loss_1_update = criterion(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = criterion(y_2[ind_1_update], t[ind_1_update])

    if noise_or_not != None:
        return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2
    else:
        return loss_1_update,loss_2_update


class Combiner:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.type = cfg.TRAIN.COMBINER.TYPE
        self.device = device
        self.epoch_number = cfg.TRAIN.MAX_EPOCH
        self.func = torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()
        if cfg.TRAIN.COMBINER.TYPE == 'coteaching':
            self.init_forget_rate()

    def init_forget_rate(self):
        # TODO: make args more flexible
        self.forget_rate = 0.2
        self.exponent = 1
        self.gradual_epoch_num = 10
        self.rate_schedule = np.ones(self.epoch_number)*self.forget_rate
        self.rate_schedule[:self.gradual_epoch_num] = np.linspace(0, self.forget_rate**self.exponent, self.gradual_epoch_num)

    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number

    def reset_epoch(self, epoch):
        self.epoch = epoch
    
    def forward(self, model, criterion, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(
            model, criterion, image, label, meta, **kwargs
        )

    def default(self, model, criterion, image, label, meta, **kwargs):
        image, label = image.to(self.device), label.to(self.device)
        output = model(image)
        loss = criterion(output, label)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]
        return loss, now_acc

    def multi_label(self,model,criterion,image,label,meta,**kwargs):
        image, label = image.to(self.device), label.to(self.device)
        output = model(image)
        loss = criterion(output, label)
        now_result = torch.sigmoid(output).ge(0.5).float()
        now_acc = (now_result == label).sum()/label.shape[0]/label.shape[1]
        return loss, now_acc

    def coteaching(self,model,criterion,image,label,meta,**kwargs):
        model1, model2 = model
        image, label = image.to(self.device), label.to(self.device)
        batch_size = image.shape[0]

        logit1=model1(image)
        # prec1 is the correct numbers/batch_size*100
        train_correct1, _ = coteaching_accuracy(logit1, label, topk=(1, 5))

        logit2 = model2(image)
        train_correct2, _ = coteaching_accuracy(logit2, label, topk=(1, 5))

        loss1, loss2 = loss_coteaching(logit1, logit2, label, criterion, self.rate_schedule[self.epoch])

        return loss1, train_correct1, loss2, train_correct2,

    def mix_up(self, model, criterion, image, label, meta, **kwargs):

        l = np.random.beta(self.alpha, self.alpha) # beta distribution

        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

        # mix up two image
        mixed_image = l * image_a + (1 - l) * image_b

        mixed_output = model(mixed_image)

        loss = l * criterion(mixed_output, label_a) + (1 - l) * criterion(mixed_output, label_b)

        now_result = torch.argmax(self.func(mixed_output), 1)
        now_acc = (
                l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )

        return loss, now_acc

    def bbn_mix(self, model, criterion, image, label, meta, **kwargs):

        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

        feature_a, feature_b = (
            model(image_a, feature_cb=True),
            model(image_b, feature_rb=True),
        )

        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay
        #l = 0.5  # fix
        #l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # cosine decay
        #l = 1 - (1 - ((self.epoch - 1) / self.div_epoch) ** 2) * 1  # parabolic increment
        #l = 1 - (self.epoch-1) / self.div_epoch  # linear decay
        #l = np.random.beta(self.alpha, self.alpha) # beta distribution
        #l = 1 if self.epoch <= 120 else 0  # seperated stage

        mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
        output = model(mixed_feature, classifier_flag=True)
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = (
                l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )

        return loss, now_acc

