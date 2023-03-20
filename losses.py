import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class MMD_loss(nn.Module):
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
	
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)
		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2)
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)
	def sample_array(self,arr1,arr2):
		if arr1.shape[0] > arr2.shape[0]:
			n = arr1.shape[0]
			ind = torch.randperm(n)[:arr2.shape[0]]
			arr1 = arr1[ind,...]
		else:
			n = arr2.shape[0]
			ind = torch.randperm(n)[:arr1.shape[0]]
			arr2 = arr2[ind,...]
		return arr1, arr2
	def forward(self, source, target):
		if source.shape[0] != target.shape[0]:
			
			source,target = self.sample_array(source,target)

			
		batch_size = int(source.size()[0])
		source = source.view(batch_size,-1)
		target = target.view(batch_size,-1)
		kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY -YX)
		return loss

class WDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        weighted diceloss
        """
        super(WDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        net_output = F.softmax(net_output,dim=1)
    
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10)
        w = w / einsum("bc->b",w).unsqueeze(1)
        w = 1 - w
        intersection: torch.Tensor = w * einsum("bcxy, bcxy->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", y_onehot))
        divided: torch.Tensor =   2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        divided = 1 - divided
        gdc = divided.mean()

        return gdc
class WJaccardLoss(nn.Module):
    def __init__(self,apply_nonlin=None):
        """
        Weighted Jaccard;
		modified from above WDiceLoss
          
        
        """
        super(WJaccardLoss, self).__init__()

        self.apply_nonlin = apply_nonlin

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        net_output = F.softmax(net_output,dim=1)
    
        
        w: torch.Tensor = (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10) 
        w = w / einsum("bc->b",w).unsqueeze(1)
        w = 1 - w
        intersection: torch.Tensor = w * einsum("bcxy, bcxy->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", y_onehot))
        divided: torch.Tensor =  1 - (einsum("bc->b", intersection)) / (einsum("bc->b", union - intersection))
        gdc = divided.mean()

        return gdc
class WBCE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        net_output = F.softmax(net_output,dim=1)
        
        w: torch.Tensor = (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10) 
        w = w / einsum("bc->b",w).unsqueeze(1)
        w = 1 - w
        w = w.mean(0)
        loss = F.cross_entropy(net_output,y_onehot,weight=w)
        
        return loss
    
def bjoern_paper_loss():
    dice = WDiceLoss()
    jaccard = WJaccardLoss()
    wbce = WBCE()
    def func(input,target):
        return dice(input,target) + (0.5*jaccard(input,target)+0.5*wbce(input,target))
    return func

def get_loss_func(loss_func):
    if loss_func == 'Dice':
        critera = smp.losses.dice.DiceLoss(mode=smp.losses.MULTICLASS_MODE,smooth=0.5)
    elif loss_func == 'WDice':
        critera = WDiceLoss()
    elif loss_func == 'WBCE':
        critera = WBCE()
    elif loss_func == 'WJaccard':
        critera = WJaccardLoss()
    elif loss_func == 'Bjoern':
        critera = bjoern_paper_loss()
    return critera