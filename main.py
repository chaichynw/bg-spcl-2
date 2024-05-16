import yaml
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataloaders.base import get_dataset
from utils.setup_utils import get_device
from trainers.offline import pretraining, eval
from trainers.online import online_learning


'''Argparse'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, default='bnci2014004_config')
parser.add_argument('--gpu_num', type=str, default='0')
parser.add_argument('--is_test', type=bool, default=False)
parser.add_argument('--online_update', type=bool, default=False)
args = parser.parse_args()

# Config setting
with open(f'configs/{aargs.config_name}.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    cfg = EasyDict(config)

cfg['is_test'] = args.is_test
cfg['online_update'] = args.online_update

# Set device
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = aargs.gpu_num
cfg['device'] = get_device(args.gpu_num)
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True


if __name__ == '__main__':

    dataset = get_dataset(cfg)

    # Offline
    if not cfg.online_update:

        if not cfg.is_test:
            pretraining(cfg, dataset)

        else:
            test_acc_dict = eval(cfg, dataset)
            print(test_acc_dict)
            print(f'Average test accuracy: {np.mean(list(test_acc_dict.values()))}')
            print(f'Std test accuracy: {np.std(list(test_acc_dict.values()))}')
    # Online
    else:
        online_learning(cfg, dataset)