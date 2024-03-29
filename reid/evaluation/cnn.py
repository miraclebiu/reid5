from __future__ import absolute_import
from collections import OrderedDict

import torch
from torch.autograd import Variable

from ..utils import to_torch


th_version = torch.__version__


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    if th_version == '0.3.0.post4':
        inputs = Variable(inputs, volatile=True)
    else:
        with torch.no_grad():
            inputs = Variable(inputs)
    if modules is None:
        # before softmax
        outputs = model(inputs)[-1]
        # # after softmax
        # outputs = model(inputs)[0]
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
