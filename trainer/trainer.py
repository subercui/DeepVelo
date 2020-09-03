import numpy as np
import torch
import dgl
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _compute_core(self, batch_data):
        if isinstance(batch_data, dgl.nodeflow.NodeFlow):
            nf = batch_data
            nf.copy_from_parent()
            nf.layers[0].data['Ux_sz'] = nf.layers[0].data['Ux_sz'].to(self.device)
            nf.layers[0].data['Sx_sz'] = nf.layers[0].data['Sx_sz'].to(self.device)
            nf.layers[-1].data['Ux_sz'] = nf.layers[-1].data['Ux_sz'].to(self.device)
            nf.layers[-1].data['Sx_sz'] = nf.layers[-1].data['Sx_sz'].to(self.device)
            target = nf.layers[-1].data['velo'].to(self.device)
            output = self.model(nf)
        else:
            data_dict = batch_data
            x_u, x_s, target = data_dict['Ux_sz'], data_dict['Sx_sz'], data_dict['velo']
            x_u, x_s, target = x_u.to(self.device), x_s.to(self.device), target.to(self.device)

            output = self.model(x_u, x_s)
        return output, target

    def _smooth_constraint_step(self):
        topG = self.config['data_loader']['args']['topG']
        neighbor_batch_ind = self.data_loader.dataset.gen_neighbor_batch(
            size=int(self.config['data_loader']['args']['batch_size']/topG))
        x_u = self.data_loader.dataset.Ux_sz[neighbor_batch_ind].to(self.device)  # (batch, genes)
        x_s = self.data_loader.dataset.Sx_sz[neighbor_batch_ind].to(self.device)
        output_neighbor = self.model(x_u, x_s)  # (batch, genes)
        self.optimizer.zero_grad()
        output_neighbor = output_neighbor.t().reshape([-1, topG])
        label_neighors = output_neighbor[:, 1:].detach()
        pivot = output_neighbor[:, 0:1]
        loss_c = torch.mean((label_neighors - pivot)**2)
        loss_c.backward()
        self.optimizer.step()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch_data in enumerate(self.data_loader):
            output, target = self._compute_core(batch_data)
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if self.config['constraint_loss']:
                self._smooth_constraint_step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_data_loader):
                output, target = self._compute_core(batch_data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
