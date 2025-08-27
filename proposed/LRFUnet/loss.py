import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss

class My_Loss(nn.Module):
    def __init__(self, num_class=3, loss_type='L1', batch_dice=None):
        super(My_Loss, self).__init__()
        self.num_classes = num_class
        self.loss_type = loss_type
        self.data_range = 1.0
        if loss_type == 'L1':
            self.loss = F.l1_loss
        elif loss_type == 'L2':
            self.loss = F.mse_loss
        elif loss_type == 'SmoothL1':
            self.loss = F.smooth_l1_loss
        elif loss_type == 'SSIM':
            self.ssim = StructuralSimilarityIndexMeasure(data_range=float(num_class))
        elif loss_type == 'SmoothL1+SSIM':
            self.SmoothL1 = F.smooth_l1_loss
            self.ssim = StructuralSimilarityIndexMeasure(data_range=float(num_class))
        elif loss_type == 'Dice+CE+SmoothL1':
            self.DC_CE = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
            self.SmoothL1 = F.smooth_l1_loss
        else:
            raise NotImplementedError("Unknown loss")
        
    def forward(self, x, y, Dice_CE_weight=None):
        device = x.device

        # 將y轉換為one-hot encoding
        y = y.long()
        y_one_hot = torch.zeros(x.size(0), self.num_classes, x.size(2), x.size(3), device=device)
        y_one_hot.scatter_(1, y, 1)

        if self.loss_type == 'SSIM':
            self.ssim = self.ssim.to(device)
            loss = 1 - self.ssim(x, y_one_hot)
        elif self.loss_type == 'SmoothL1+SSIM':
            self.ssim = self.ssim.to(device)
            loss = self.SmoothL1(x, y_one_hot, reduction='none') + (1 - self.ssim(x, y_one_hot))
        elif self.loss_type == 'Dice+CE+SmoothL1':
            if Dice_CE_weight is None:
                loss = self.DC_CE(x, y) + self.SmoothL1(x, y_one_hot, reduction='none')
            else:
                print(f"Dice_CE_weight in My_Loss: {Dice_CE_weight}")
                loss = Dice_CE_weight * self.DC_CE(x, y) + (1 - Dice_CE_weight) * self.SmoothL1(x, y_one_hot, reduction='none')
        else:
            loss = self.loss(x, y_one_hot, reduction='none')
        return loss.mean()
    
class My_MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(My_MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        #self.Dice_CE_weights = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
        self.Dice_CE_weights = None
        #self.Dice_CE_weights = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), requires_grad=True).cuda()
        #print("My_MultipleOutputLoss2.parameters()", list(self.parameters()))

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        if self.Dice_CE_weights is None:
            l = weights[0] * self.loss(x[0], y[0], None)
        else:
            l = weights[0] * self.loss(x[0], y[0], self.Dice_CE_weights[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                if self.Dice_CE_weights is None:
                    l += weights[i] * self.loss(x[i], y[i], None)
                else:
                    #print(f'self.Dice_CE_weights[{i}]: {self.Dice_CE_weights[i]}')
                    #print(f'self.Dice_CE_weights[{i}].grad: {self.Dice_CE_weights[i].grad}')
                    #print(f'self.Dice_CE_weights[{i}].requires_grad: {self.Dice_CE_weights[i].requires_grad}')
                    l += weights[i] * self.loss(x[i], y[i], self.Dice_CE_weights[i])
        return l