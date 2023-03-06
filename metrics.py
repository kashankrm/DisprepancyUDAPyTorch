from PIL import Image
import random
import segmentation_models_pytorch as smp
def dummy_metric(*args):
    return 1.0
def iou_metric_fg(*args):
    #(f"{dt}__{mt}",self.train_it,inputs,outputs,labels,im_name)
    pred = args[3]
    labels = args[4]
    output = pred.argmax(1)
    tp, fp, fn, tn = smp.metrics.get_stats(output, labels, mode='multilabel', threshold=0.5)

    # then compute metrics with required reduction (see metric docs)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return iou_score
def iou_metric_bg(*args):
    #(f"{dt}__{mt}",self.train_it,inputs,outputs,labels,im_name)
    pred = args[3]
    labels = args[4]
    pred = 1-pred
    labels = 1-labels
    output = pred.argmax(1)
    tp, fp, fn, tn = smp.metrics.get_stats(output, labels, mode='multilabel', threshold=0.5)

    # then compute metrics with required reduction (see metric docs)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    return iou_score

def metric_wrapper(func,defaults,arglist):

    def wrapped_func(*args):
        true_args = [args[i] for i in arglist]
        return func(*defaults,*true_args)
    return wrapped_func