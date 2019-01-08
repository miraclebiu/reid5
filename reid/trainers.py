from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import pdb
from .evaluation import accuracy
from .loss import support_loss
from .evaluation import AverageMeter

th_version = torch.__version__


class Trainer(object):
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs

        # inputs = [Variable(imgs)]
        inputs = Variable(imgs.cuda())
        # pdb.set_trace()
        targets = Variable(pids.cuda())
        return inputs, targets

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            outputs = self.model(inputs)
            loss, prec = self.criterion(outputs, targets)
            if th_version == '0.3.0.post4':
                losses.update(loss.data[0], targets.size(0))
            else:
                losses.update(loss.item(), targets.size(0))
            precisions.update(prec, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

