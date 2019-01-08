import torch
import torch.nn.functional as F
import numpy as np
import pdb

qfea = torch.rand(32,512)
gfea = torch.rand(100,512)
qfea = F.normalize(qfea)
gfea = F.normalize(gfea)

q2g = torch.mm(qfea, gfea.t())


topk_num = 50
topk_dist, topk_ind = q2g.topk(topk_num)

temperature = 0.1
topk_dist = topk_dist / temperature

weight = F.softmax(topk_dist,1)

m = qfea.size(0)
sft_qfea = []
for i in range(m):
    temp_fea = torch.index_select(gfea,0,topk_ind[i])
    temp_weight = weight[i].unsqueeze(1)
    new_fea = temp_weight * temp_fea
    sft_qfea.append(new_fea.sum(0))
pdb.set_trace()
qfea2 = torch.stack(sft_qfea)


