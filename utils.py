import albumentations as albu
def combine_val_ret(val_ret):
    def get_values(k,data):
        vals = [vd[k] for vd in data]
        return vals
    losses = {}
    metrics = {}
    val_loss = val_ret['losses']
    val_metric = val_ret['metrics']
    loss_keys = val_loss[0].keys()
    for lk in loss_keys:
        vals = get_values(lk,val_loss)
        losses[lk] = sum(vals)/len(vals)
    metric_keys = val_metric[0].keys()
    for mk in metric_keys:
        vals = get_values(mk,val_metric)
        metrics[mk] = sum(vals)/len(vals)
    return losses,metrics
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
def to_tensor(x, **kwargs):
    return (x.transpose(2, 0, 1) if len(x.shape) > 2 else x).astype('float32')