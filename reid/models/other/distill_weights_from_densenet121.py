import torch
from torchvision import models
from densenet_base import DenseNet_Biu
import pdb

dn121 = models.densenet121(pretrained=True)
dn121_weights = dn121.state_dict()
dn121_weights_keys =  dn121_weights.keys()

dn_biu = DenseNet_Biu()
dn_biu_weights = dn_biu.state_dict()
dn_biu_weights_keys = dn_biu_weights.keys()
keep_keys = {
          'features.conv0.weight':'features.conv0.weight', 
          'features.norm0.weight':'features.norm0.weight', 
          'features.norm0.bias':'features.norm0.bias', 
          'features.norm0.running_mean':'features.norm0.running_mean', 
          'features.norm0.running_var':'features.norm0.running_var', 
}
new_keys={
          'layer1':'features.denseblock1',
          'trans1':'features.transition1',
          'layer2':'features.denseblock2',
          'trans2':'features.transition2',
          'layer3':'features.denseblock3',
          'trans3':'features.transition3',
          'layer4':'features.denseblock4',
          'features_norm5':'features.norm5',
}
already_handle = []
for dn_biu_key in dn_biu_weights:
     prefix = dn_biu_key.split('.')[0]
     if prefix =='classifier':
          continue
     if prefix in new_keys.keys():
          ori_key = dn_biu_key.replace(prefix,new_keys[prefix],1)
     else:
          ori_key = dn_biu_key

     if ori_key.split('.')[0]=='feat':
          print('Already convert the weight before feat.weight')
          break
     print('ori_key: ',ori_key)
     print('biu_key: ',dn_biu_key)
     dn_biu_weights[dn_biu_key] = dn121_weights[ori_key]
     already_handle.append(dn_biu_key)

torch.save(dn_biu_weights,'dn_biu.pkl')
# pdb.set_trace()