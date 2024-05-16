import yaml
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloaders.base import get_dataset
from utils.setup_utils import get_device
from offline import pretraining, eval
from online import online_learning


'''Argparse'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='moabb_config')
parser.add_argument('--gpu_num', type=str, default='0')
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--online_update', type=bool, default=False)
aargs = parser.parse_args()

# Config setting
with open(f'configs/{aargs.config_name}.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)

# Update args
args.is_test = aargs.is_test
args.online_update = aargs.online_update

# Set device
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = aargs.gpu_num
args['device'] = get_device(aargs.gpu_num)
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True


if __name__ == '__main__':

    dataset = get_dataset(args)

    # Offline
    if not args.online_update:

        if not args.is_test:
            pretraining(args, dataset)

        else:
            test_acc_dict = eval(args, dataset)
            print(test_acc_dict)
            print(f'Average test accuracy: {np.mean(list(test_acc_dict.values()))}')
            print(f'Std test accuracy: {np.std(list(test_acc_dict.values()))}')
    # Online
    else:
        online_learning(args, dataset)