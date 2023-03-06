

from datetime import datetime
import random
import torch
import copy
import torch.nn.functional as F
from utils import combine_val_ret
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self,
                model,
                opt,
                iterations,
                device,
                logger:SummaryWriter,
                val_img_save=True,
                val_every_it:int=-1,
                print_train_every_iter = 1
                ) -> None:
        self.model = model.to(device)
        self.logger = logger
        self.val_img_save = val_img_save
        self.device = device
        self.opt = opt
        self.iterations = iterations
        self.val_every_it = val_every_it
        self.print_train_every_iter = print_train_every_iter
        self.datasets = {}
        self.criteria = {}
        self.metric = {}
        self.phases = {}
        self.train_it = None
        self.output_buffer = {}

    def add_dataset(self,name,data):
        self.datasets[name] = {
            'data':data,
            'metrics': [],
            }

    def add_criteria(self,name,criteria):
        self.criteria[name] = criteria

    def add_metric(self,name,metric_func):
        self.metric[name] = metric_func

    def assign_criteria_to_dataset(self,phase_name,dt_name,cr_name,cr_weight=1.0):
        phase = self.phases.get(phase_name,[])
        phase.append((dt_name,cr_name,cr_weight))
        self.phases[phase_name] = phase

    def assign_metric_to_dataset(self,dt_name,mt_name):
        dt_metrics = self.datasets[dt_name].get('metrics',[])
        dt_metrics.append(mt_name)
        self.datasets[dt_name]['metrics'] = dt_metrics

    def get_batch(self,dt_name):
        dt_loader = self.datasets[dt_name].get('dt_loader')
        if dt_loader is None:
            dt_loader = iter(copy.deepcopy(self.datasets[dt_name]['data']))
        while True:
            try:
                inputs, labels,_, name = next(dt_loader)
                break
            except StopIteration as e:
                dt_loader = iter(copy.deepcopy(self.datasets[dt_name]['data']))
        self.datasets[dt_name]['dt_loader'] = dt_loader
        return inputs.to(self.device),labels.to(self.device), name
    def calculate_loss(self,cr,outputs,labels):
        if cr is None:
            return 0.0
        
        return self.criteria[cr](outputs, labels) 
    
    def calculate_metrics(self,mt,dt,inputs,outputs,labels,im_name):
        return self.metric[mt](f"{dt}__{mt}",self.train_it,inputs,outputs,labels,im_name)
    
    def model_forward(self,x):
        return self.model(x)
    
    def save_images(self,dt,inputs,labels,outputs):
        r_idx = random.randint(0,inputs.shape[0]-1)
        self.logger.add_image(f"val_inp_{dt}",inputs[r_idx],global_step=self.train_it)
        self.logger.add_image(f"val_label_{dt}",labels[r_idx,None],global_step=self.train_it)
        self.logger.add_image(f"val_output_{dt}",outputs[r_idx].argmax(dim=0)[None,...],global_step=self.train_it)
        # self.logger.add_scalar("val_im_name",im_name[r_idx],global_step=self.train_it)
    
    def run_step(self,phase):
        losses = {}
        metrics = {}
        dt_cr_pairs = self.phases[phase]
        #clear output buffer
        self.output_buffer = {}
        for dt,cr,cr_wth in dt_cr_pairs:
            if dt in self.output_buffer:
                outputs, labels = self.output_buffer[dt]
            else:
                inputs, labels, im_name = self.get_batch(dt)    
                outputs = self.model_forward(inputs)
                self.output_buffer[dt] = (outputs,labels)
                if phase == 'val' and self.val_img_save:
                    self.save_images(dt,inputs,labels,outputs)
                    
                with torch.no_grad():
                    for mt in self.datasets[dt]['metrics']:
                        # add things to this if it is needed in one of metric functions
                        metrics[f"{dt}__{mt}"] = self.calculate_metrics(mt,dt,inputs,outputs,labels,im_name)
                # Calculate the loss
                
            loss = self.calculate_loss(cr, outputs, labels) 
            losses[f"{dt}__{cr}"] = cr_wth*loss
        
        return losses, metrics
    def optimizer_step(self,):
        self.opt.step()
        return 
    def model_loop(self,phase,iterations,train=False):
        def single_iter():                
            if train:
            # Clear the gradients
                self.opt.zero_grad()
            
            # Make a forward pass
            losses,metrics = self.run_step(phase)
            if train:
                loss = sum(list(losses.values()))        
                # Backpropagate
                loss.backward()
                
                # Update the weights
                self.optimizer_step()
            return losses,metrics
        
        
        losses, metric_lst = [],[]
        val_losses, val_metric_lst = [],[]
        
        for i in range(iterations):
            
            if train:
                loss,metric = single_iter()    
            else:
                with torch.no_grad():
                    loss,metric = single_iter()
            loss = {k:(v.item() if not (isinstance(v,float) or isinstance(v,int)) else v) for k,v in loss.items()}
            losses.append(loss)
            metric_lst.append(metric)
            if phase == 'train':
                if (i+1)%self.print_train_every_iter == 0:
                    print(f"{i}/{self.iterations} losses {loss}, metrics {metric}")
                self.train_it = i
                for k,v in loss.items():
                    self.logger.add_scalar(f"loss_{k}",v,global_step=i)
                for k,v in metric.items():
                    self.logger.add_scalar(f"metric_{k}",v,global_step=i)
                if self.val_every_it>0 and (i+1)% self.val_every_it==0:
                    val_ret = self.validate()
                    val_loss, val_metric = combine_val_ret(val_ret)
                    print(f"{i}/{self.iterations} val_losses {val_loss}, val_metrics {val_metric}")
                    val_losses.append(val_loss)
                    val_metric_lst.append(val_metric)
                    for k,v in val_loss.items():
                        self.logger.add_scalar(f"val_loss_{k}",v,global_step=i)
                    for k,v in val_metric.items():
                        self.logger.add_scalar(f"val_metric_{k}",v,global_step=i)
                    
        # kinda hacky way to check what training iteration it is
        self.train_it = None
        ret = {
            'losses' : losses,
            'metrics' : metric_lst,

        }
        if phase == 'train':
            ret['val_losses'] = val_losses
            ret['val_metrics'] = val_metric_lst
        return ret
    
    def train(self):
        return self.model_loop('train',self.iterations,True)
    def validate(self):
        if 'val' in self.phases:
            val_dts = [p[0]  for p in self.phases['val']]
            
            max_iterations = max([len(self.datasets[dt]['data']) for dt in val_dts])
            max_iterations = min(max_iterations,100)
            self.model.eval()
            ret  = self.model_loop('val',max_iterations,False)
            self.model.train()
            return ret
class DiscrepencyDATrainer(Trainer):
    def __init__(self, 
                model, 
                opt, 
                iterations, 
                device, 
                logger: SummaryWriter, 
                val_img_save=True, 
                val_every_it: int = -1,
                print_train_every_iter = 1
                ) -> None:
        super().__init__(
                        model, 
                        opt, 
                        iterations, 
                        device, 
                        logger, 
                        val_img_save, 
                        val_every_it,
                        print_train_every_iter)
        self.disp_criteria_pairs = []

    # def add_criteria(self, name:str, criteria):
    #     self.criteria[name] = {
    #         'type': "sup",
    #         'func': criteria
    #     }
    #     return
    # def add_dicrepency_criteria(self,name:str,criteria,level:str):
    #     self.criteria[name] = {
    #         'type': "disc",
    #         'func': criteria,
    #         'level': level
    #     }
    def assign_disp_criteria_to_dataset(self,dt1_name,dt2_name,cr_name,level,cr_weight=1.0):
        self.disp_criteria_pairs.append((dt1_name,dt2_name,cr_name,level,cr_weight))
        
    #TODO: figure out a better way to deal with this outputs being a tuple problem
    def calculate_metrics(self, mt, dt, inputs, outputs, labels, im_name):
        outputs = outputs[0]
        return super().calculate_metrics(mt, dt, inputs, outputs, labels, im_name)
    
    def calculate_loss(self, cr, outputs, labels):
        return super().calculate_loss(cr,outputs[0],labels)
    
    def save_images(self, dt, inputs, labels, outputs):
        return super().save_images(dt, inputs, labels, outputs[0])
    
    def calculate_disc_loss(self, cr, outputs, labels):
        return super().calculate_loss(cr,outputs,labels)
    
    def extract_lvl_features(self,arr,lvl):
        if lvl == 'output':
            feat = arr[0]
        else:
            lvl = int(lvl)
            feat = arr[1][lvl]
        return feat
    def run_step(self, phase):
        #go through all supervised training criterias first
        losses,metrics = super().run_step(phase)

        for dt1,dt2,cr,lvl,cr_wth in self.disp_criteria_pairs:
            if dt1 in self.output_buffer:
                outputs_dt1, labels_dt1 = self.output_buffer[dt1]
            else:
                inputs_dt1, labels_dt1, _ = self.get_batch(dt1)    
                outputs_dt1 = self.model_forward(inputs_dt1)
                self.output_buffer[dt1] = (outputs_dt1,labels_dt1)
            if dt2 in self.output_buffer:
                outputs_dt2, labels_dt2 = self.output_buffer[dt2]
            else:
                inputs_dt2, labels_dt2, _ = self.get_batch(dt2)    
                outputs_dt2 = self.model_forward(inputs_dt2)
                self.output_buffer[dt2] = (outputs_dt2,labels_dt2)
            
            feat1, feat2 = self.extract_lvl_features(outputs_dt1,lvl), self.extract_lvl_features(outputs_dt2,lvl)
            loss = self.calculate_disc_loss(cr, feat1, feat2) 
            losses[f"{dt1}__{dt2}__{cr}"] = cr_wth*loss
        return losses, metrics
    
    def model_forward(self, x):
        return self.model.forward_extra(x,out_features=True,out_decoder_output=False)

class AdversarialDATrainer(DiscrepencyDATrainer):
    def __init__(
            self, 
            model, 
            opt, 
            iterations, 
            device, 
            logger: SummaryWriter, 
            val_img_save=True, 
            val_every_it: int = -1, 
            print_train_every_iter=1
            
            ) -> None:
        super().__init__(
                model, 
                opt, 
                iterations, 
                device, 
                logger, 
                val_img_save, 
                val_every_it, 
                print_train_every_iter
                )
        self.adv_model = {}
        self.adv_opt = {}
        self.adv_criteria_pairs = []
    def set_adv_training(self,adv,train=False):
        for param in self.adv_model[adv].parameters():
            param.requires_grad = train
        
    def add_adv_model(self,model_name,model,opt):
        self.adv_model[model_name] = model.to(self.device)
        self.adv_opt[model_name] = opt
        self.set_adv_training(model_name,False)
        
    def assign_adv_criteria_to_dataset(self,dt1_name,dt2_name,adv_model_name,cr_name,level,cr_weight=0.001):
        self.adv_criteria_pairs.append((dt1_name,dt2_name,adv_model_name,cr_name,level,cr_weight))

    def run_step(self, phase):
        #first let discrepency loss do its training
        losses,metrics = super().run_step(phase)
        for dt1,dt2,adv,cr,lvl,cr_wth in self.adv_criteria_pairs:
            if dt1 in self.output_buffer:
                outputs_dt1, labels_dt1 = self.output_buffer[dt1]
            else:
                inputs_dt1, labels_dt1, _ = self.get_batch(dt1)    
                outputs_dt1 = self.model_forward(inputs_dt1)
                self.output_buffer[dt1] = (outputs_dt1,labels_dt1)
            if dt2 in self.output_buffer:
                outputs_dt2, labels_dt2 = self.output_buffer[dt2]
            else:
                inputs_dt2, labels_dt2, _ = self.get_batch(dt2)    
                outputs_dt2 = self.model_forward(inputs_dt2)
                self.output_buffer[dt2] = (outputs_dt2,labels_dt2)
            
            feat1, feat2 = self.extract_lvl_features(outputs_dt1,lvl), self.extract_lvl_features(outputs_dt2,lvl)
            loss = self.calculate_adv_loss(cr, adv, feat2) 
            losses[f"{dt1}__{dt2}__{adv}"] = self.run_step_adv(feat1,feat2,adv,cr)
            losses[f"{dt1}__{dt2}__{cr}"] = cr_wth*loss

        return losses,metrics
    def run_step_adv(self,feat1,feat2,adv,cr):
        criteria = self.criteria[cr]
        feat1 = feat1.detach()
        feat2 = feat2.detach()
        D_out1 = self.adv_model[adv](F.softmax(feat1,dim=1))
        D_out2 = self.adv_model[adv](F.softmax(feat2,dim=1))
        loss = criteria(D_out1,torch.FloatTensor(D_out1.data.size()).fill_(0).to(self.device))
        loss = loss + criteria(D_out2,torch.FloatTensor(D_out2.data.size()).fill_(1).to(self.device))
        return loss
    def optimizer_step(self):
        super().optimizer_step()
        for adv in self.adv_model:
            self.set_adv_training(adv,True)
            self.adv_opt[adv].step()
            self.set_adv_training(adv,False)
        return 
    def calculate_adv_loss(self, cr, adv, feat2):
        criteria = self.criteria[cr]
        D_out = self.adv_model[adv](F.softmax(feat2,dim=1))
        return criteria(D_out,torch.FloatTensor(D_out.data.size()).fill_(0).to(self.device))
        