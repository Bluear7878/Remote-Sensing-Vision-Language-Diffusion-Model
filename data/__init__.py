'''create dataset and dataloader'''
import logging
from re import split

import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase, img_mode = "dir"):
    '''create dataset'''
    mode = dataset_opt['mode']

    if mode == 'single':
        from data.single_image_dataset import SingleImageDataset as D
        dataset = D(hr_path=dataset_opt['hr_path'],
                    sr_path=dataset_opt.get('sr_path', None),
                    lr_path=dataset_opt.get('lr_path', None),
                    need_LR=dataset_opt.get('need_LR', False),
                    split=phase)
    else :
        from data.LRHR_dataset import LRHRDataset as D
        dataset = D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    l_resolution=dataset_opt['l_resolution'],
                    r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LR=(mode == 'LRHR')
                    )
    logger = logging.getLogger('base')
    #logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,dataset_opt['name']))
    return dataset

def create_dataset2(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset2 import LRHRDataset2 as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=True
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
