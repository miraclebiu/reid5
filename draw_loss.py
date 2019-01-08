import os
import pdb
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']

import numpy as np
from scipy.interpolate import spline, interp1d


log_txt = './log.txt'

f = open(log_txt,'r')
content = f.readlines()
f.close()

Loss_patt = re.compile("(Loss )(\d+\.\d+)")
Epoch_patt = re.compile("(Epoch: \[)([0-9]*)(\])")
mAP_patt = re.compile("(Mean AP: )(\d+\.\d+)(%)")
curr_epoch = 0 
curr_loss_sum = 0
count = 0
loss_curve = []
mAP = []
for each_line in content:
	epoch_match = Epoch_patt.search(each_line)
	if epoch_match:
		epoch_num = int(epoch_match.group(2))
		curr_loss  = float(Loss_patt.search(each_line).group(2))
		if epoch_num != curr_epoch:
			mean_loss = curr_loss_sum / count
			loss_curve.append(mean_loss)
			# print('epoch: {}, loss: {:.3f}'.format(epoch_num, mean_loss))
			curr_epoch = epoch_num
			curr_loss_sum = 0
			count = 0
			# pdb.set_trace()
		else:
			count +=1
			curr_loss_sum += curr_loss
	mAP_match = mAP_patt.search(each_line)
	if mAP_match:
		mAP_value = float(mAP_match.group(2))
		print(mAP_value)
		mAP.append(mAP_value)
		# pdb.set_trace()
step_epoch = 70
evaluate_steps =[i for i in range(1,step_epoch*2) if i%10==0] + [i for i in range(step_epoch*2,len(loss_curve)+1,3)]
converge_epoch=90

x = range(converge_epoch)
y = loss_curve[:converge_epoch]

x_smooth = np.linspace(0,converge_epoch-1,1000)
smooth_func = interp1d(x,y,kind='cubic')
y_smooth = smooth_func(x_smooth)

plt.figure(0) #创建绘图对象
plt.ylim(0.1, 0.45)  # 限定纵轴的范围
plt.plot(x_smooth, y_smooth, 'b')
plt.legend()  # 让图例生效
plt.xlabel(u"训练轮数") #X轴标签
plt.axvline(step_epoch,ls="--",color="r")
plt.annotate(u"lr_step",xy=(step_epoch-12.5,0.8),color="r",size=15)
plt.ylabel("损失函数值") #Y轴标签
plt.title("交叉熵损失函数前100轮训练数值变化曲线") #标题
plt.subplots_adjust( wspace = 0, hspace = 0 )  
plt.savefig("loss_curve_converge.jpg") #保存图 
plt.close(0)


x = range(len(loss_curve))
y = loss_curve
x_smooth = np.linspace(0,len(loss_curve)-1,500)
smooth_func = interp1d(x,y,kind='cubic')
y_smooth = smooth_func(x_smooth)

plt.figure(1) #创建绘图对象
plt.ylim(0.1, 0.8)  # 限定纵轴的范围
plt.plot(x_smooth, y_smooth, 'b')
plt.legend()  # 让图例生效
plt.xlabel(u"训练轮数") #X轴标签

plt.axvline(step_epoch,ls="--",color="r")
plt.annotate(u"lr_step",xy=(step_epoch-12.5,0.9),color="r",size=15)

plt.axvline(step_epoch*2,ls="--",color="r")
plt.annotate(u"lr_step",xy=(step_epoch*2-12.5,0.9),color="r",size=15)

plt.axvline(step_epoch*3,ls="--",color="r")
plt.annotate(u"lr_step",xy=(step_epoch*3-12.5,0.9),color="r",size=15)

plt.ylabel("损失函数值") #Y轴标签
plt.title("交叉熵损失函数训练数值变化曲线") #标题
# plt.show()
plt.savefig("loss_curve.jpg") #保存图 
plt.close(1)


x = range(len(loss_curve))
y = loss_curve
x_smooth = np.linspace(0,len(loss_curve)-1,1000)
smooth_func = interp1d(x,y,kind='cubic')
y_smooth = smooth_func(x_smooth)


step_epoch = 70
evaluate_steps =[i for i in range(1,step_epoch*2) if i%10==0] + [i for i in range(step_epoch*2,len(loss_curve)+1,3)]
x = evaluate_steps
y = mAP[:-1] # the last mAP evaluation is for showing the best mAP, not training evaluation

x_smooth = np.linspace(10,evaluate_steps[-1]-1,200)
smooth_func = interp1d(x,y,kind='cubic')
y_smooth = smooth_func(x_smooth)

plt.figure(2) #创建绘图对象
plt.plot(x_smooth, y_smooth, 'b')
plt.legend()  
plt.xlabel(u"训练轮数") #X轴标签

plt.axvline(step_epoch,ls="--",color="r")
plt.annotate(u"lr_step",xy=(step_epoch-12.5,0.8),color="r",size=15)

plt.axvline(step_epoch*2,ls="--",color="r")
plt.annotate(u"lr_step",xy=(step_epoch*2-12.5,0.8),color="r",size=15)

plt.axvline(step_epoch*3,ls="--",color="r")
plt.annotate(u"lr_step",xy=(step_epoch*3-12.5,0.8),color="r",size=15)

plt.ylabel("平均精度均值(%)") #Y轴标签
plt.title("平均精度均值变化曲线") #标题
# plt.show()
plt.savefig("mAP_curve.jpg") #保存图
plt.close(1) 






