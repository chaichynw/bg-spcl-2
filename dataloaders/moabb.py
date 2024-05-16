import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch 
import torch.utils.data as Data

from bg_spcl.bg import compute_bg_scores


def BNCI2014004(args):

    X = np.load(f'data/{args.data}/X.npy')
    y = np.load(f'data/{args.data}/labels.npy')
    meta = pd.read_csv(f'data/{args.data}/meta.csv')

    train_indices = [i for i, (session, subject) in enumerate(zip(meta['session'], meta['subject'])) if session == '1train' and subject == args.target_subject+1]
    test_indices = [i for i, (session, subject) in enumerate(zip(meta['session'], meta['subject'])) if session == '0train' and subject == args.target_subject+1]

    if not args.is_test:
        print('\n Train dataset is loaded \n')
        X = X[train_indices]
        y = y[train_indices]
    else:
        print('\n Test dataset is loaded \n')
        X = X[test_indices]
        y = y[test_indices]

    X = X[:, :, :, np.newaxis]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    c = compute_bg_scores(args, X) 

    y = torch.from_numpy(y.reshape(-1, )).to(torch.long)
    X = torch.from_numpy(X).to(torch.float32)
    c = torch.tensor(c)

    dataloader = {}   
    if args.model != 'EEGITNet':
        X = X.permute(0, 3, 1, 2)
     
    X, y = X.to(args.device), y.to(args.device)

    data = Data.TensorDataset(X, y, c)
        
    sample = {'data': X, 'label': y, 'attention': c}

    return data

