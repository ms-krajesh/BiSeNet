* 看data是否正确，为啥输出变在(360, 640) 了
* 看是否是训练20个类，但是预测时只剩下19个了


#===========
baseline
#===========
1.
momentum = 0.9
weight_decay = 1e-4
lr_start = 1e-3
max_iter = 11000
power = 0.9
warmup_steps = 2000
warmup_start_lr = 1e-6

mIOU = 34.56

2. paper里面的train 参数，
11000iter无warmup: 37.67
31000iter无warmup:
11000iter无warmup(修正warmup step): 


1. 使用auxiliary的loss

2. eval 时像论文那样使用1024x1024

2. 如果不收敛就先warmup一下再弄
