import torch
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import pdb
import copy
from resnet_mgn import resnet50_mgn,resnet101_mgn


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

def copy_weight_for_branches(model_name, branch_num=3):
	model_weight = model_zoo.load_url(model_urls[model_name])
	model_keys = model_weight.keys()

	new_model = eval(model_name+'_mgn()')
	new_model_weight = new_model.state_dict()
	new_model_keys = new_model_weight.keys()
	# new_model_weight = copy.deepcopy(model_weight)
	handled =[]
	for block in new_model_keys:
		print(block)
		prefix = block.split('.')[0]
		ori_prefix = prefix.split('_')[0]
		suffix = block.split('.')[1:]
		if(ori_prefix == 'layer3') or (ori_prefix =='layer4'):
			for i in range(0,branch_num):
				ori_key_to_join = [ori_prefix] + suffix
				ori_key = '.'.join(ori_key_to_join)
				if ori_key in model_keys:
					print(new_model_weight[block].size(),model_weight[ori_key].size())
					new_model_weight[block] = model_weight[ori_key]
				else:
					continue
				# pdb.set_trace()
				# handled.append(ori_key)
		elif ori_prefix in model_keys:
			new_model_weight[block] = model_weight[ori_key]
	# pdb.set_trace()
	save_name = model_name + '_mgn.tar'
	torch.save(new_model_weight,save_name)
	pdb.set_trace()

if __name__ == '__main__':
	copy_weight_for_branches('resnet50')



