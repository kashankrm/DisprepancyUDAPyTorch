from torch import nn
import segmentation_models_pytorch as smp
from typing import Optional, List, Union

def add_feature_extractor(model):
    # adds a forward hook to encoder of the model
    # so that we can save its output which will 
    # then be returned in foward_extra function
     
    def caching_encoder_hook(self, input,output):
        self.encoder_cache=output
        
    model.encoder.register_forward_hook(caching_encoder_hook)
    # kinda hacky way to add an extra function to the model
    def forward_extra(x,out_features=False):
        ret = (model.forward(x),)
        features = model.encoder.encoder_cache
        if out_features:
            ret = (*ret,features)
        return ret
    model.forward_extra = forward_extra
    return
class UnetPlusPlus(smp.UnetPlusPlus):
    def __init__(self, 
                 encoder_name: str = "resnet34", 
                 encoder_depth: int = 5, 
                 encoder_weights: Optional[str] = "imagenet", 
                 decoder_use_batchnorm: bool = True, 
                 decoder_channels: List[int] = (256, 128, 64, 32, 16), 
                 decoder_attention_type: Optional[str] = None,
                 in_channels: int = 3, 
                 classes: int = 1, 
                 activation: Optional[Union[str, callable]] = None, 
                 aux_params: Optional[dict] = None
                 ):
        
        super().__init__(encoder_name, encoder_depth, encoder_weights, decoder_use_batchnorm, decoder_channels, decoder_attention_type, in_channels, classes, activation, aux_params)
    def forward_extra(self,x,out_features=False,out_decoder_output=False):
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            ret = (masks, labels)
        ret = (masks,)
        if out_features:
            ret = (*ret,features)
        if out_decoder_output:
            ret = (*ret,decoder_output)

        return ret


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.interp = nn.Upsample(size=(36, 36), mode='bilinear', align_corners=True)


    def forward(self, x):
        if x.shape[-1] < 36:
            x = self.interp(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x
def get_encoder_out_shape(args):
    encoder_name = args.encoder
    if encoder_name == "densenet201":        
        return {

            "output":args.num_classes,
            "0":3,
            "1":64,
            "2":256,
            "3":512,
            "4":1792,
            "5":1920,
            }
    elif encoder_name == "densenet161":
        return {
            "output":args.num_classes,
            "0":3,
            "1":96,
            "2":384,
            "3":768,
            "4":2112,
            "5":2208,
            }
    elif encoder_name == "densenet121":
        return {
            "output":args.num_classes,
            "0":3,
            "1":64,
            "2":256,
            "3":512,
            "4":1024,
            "5":1024,
            }
    elif encoder_name == "resnet18":
        return {
            "output":args.num_classes,
            "0":3,
            "1":64,
            "2":64,
            "3":128,
            "4":256,
            "5":512,
            }
    elif encoder_name == "resnet18":
        return {
            "output":args.num_classes,
            "0":3,
            "1":64,
            "2":64,
            "3":128,
            "4":256,
            "5":512,
            }
    elif encoder_name == "resnet50":
        return {
            "output":args.num_classes,
            "0":3,
            "1":64,
            "2":256,
            "3":512,
            "4":1024,
            "5":2048,
            }
    elif encoder_name == "resnet152":
        return {
            "output":args.num_classes,
            "0":3,
            "1":64,
            "2":256,
            "3":512,
            "4":1024,
            "5":2048,
            }


