from models.backbones.EEGNet import EEGNet
from models.backbones.ShallowConvNet import ShallowConvNet
from models.backbones.DeepConvNet import DeepConvNet
from models.backbones.EEGTransformer import EEGTransformer
from models.backbones.EEGITNet import EEGITNet
from models.backbones.EEGResNet import EEGResNet

model_list = {
    'EEGNet': EEGNet,
    'ShallowConvNet': ShallowConvNet,
    'DeepConvNet': DeepConvNet,
    'EEGTransformer': EEGTransformer,
    'EEGITNet': EEGITNet,
    'EEGResNet': EEGResNet
}