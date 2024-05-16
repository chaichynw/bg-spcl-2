from dataloaders.moabb import BNCI2014004


def get_dataset(args, is_test):
    
    args['is_test'] = True if is_test else False

    if args.data == 'BNCI2014004':
        args.paradigm = 'MI'
        args.num_subjects = 9
        args.num_classes = 2
        args.sampling_rate = 250
        args.num_channels = 3
        args.trial_len = 4 
        args.temporal_size = 1126
        args.feature_deep_dim = 560

        dataloader = BNCI2014004(args)

    # elif args.data == 'BNCI2015001':
    #     args.paradigm = 'MI'
    #     args.num_subjects = 12
    #     args.sampling_rate = 512
    #     args.num_classes = 2
    #     args.num_channels = 13
    #     args.trial_len = 5
    #     args.temporal_size = 2561
    #     args.feature_deep_dim = 640

    #     dataloader = BNCI2015001(args)
    
    # elif args.data == 'Zhou2016':
    #     args.paradigm = 'MI'
    #     args.num_subjects = 4
    #     args.sampling_rate = 250
    #     args.num_classes = 3
    #     args.num_channels = 14
    #     args.trial_len = 5
    #     args.temporal_size = 1251
    #     args.feature_deep_dim = 496

    #     dataloader = Zhou2016(args)

    return dataloader