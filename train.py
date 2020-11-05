import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer, AdaptiveTrainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    if config['arch']['type'] in ['VeloGCN', 'VeloGIN']:
        model = config.init_obj('arch', module_arch, g=data_loader.dataset.g)
    else:
        model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    if config['adaptive_train']:
        adaptive_trainer = AdaptiveTrainer(model, criterion, metrics, optimizer,
                                           config=config,
                                           data_loader=data_loader,
                                           valid_data_loader=valid_data_loader,
                                           lr_scheduler=lr_scheduler
                                           )
        adaptive_trainer.train()

    # evaluate all and return the velocity matrix (1720, 1448)
    config_copy = config['data_loader']['args'].copy()
    config_copy.update(
            shuffle=False,
            training=False
        )
    eval_loader = getattr(module_data, config['data_loader']['type'])(**config_copy)
    model.eval()
    velo_mat = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_loader):
            output, _ = trainer._compute_core(batch_data)
            velo_mat.append(output.cpu().data)
        velo_mat = np.concatenate(velo_mat, axis=0)
    print('velo_mat shape:', velo_mat.shape)
    np.savez(f"./data/{config['online_test']}", velo_mat=velo_mat)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--dd', '--data_dir'], type=str, target='data_loader;args;data_dir'),
        CustomArgs(['--ng', '--n_genes'], type=int, target='arch;args;n_genes'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--ot', '--online_test'], type=str, target='online_test')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
