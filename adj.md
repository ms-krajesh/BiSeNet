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
21000iter无warmup: 
11000iter无warmup(修正warmup step): 38.18

1000/11000: 38.09


1. 使用auxiliary的loss
aux_loss加3x3conv，使用feat16和32: 35.01
使用8和16的feat加aux_loss

2. ffm的conv_bn_relu的stride改成2: 31.72

3. 最后输出的net，再加一个1x1的conv，stride=1，通道不变: 28.63

4. 训练时只使用其中的19个id，让训练和测试时使用相同的id: 41.97

5. 不加auxiliary: 47.9，看来加auxiliary是有技巧的。

6. 调整sp和cp的输出通道数，让两个均匀一点: 
* sp变成64,256,512: 47.47
* 把cp变成FPN这种: 47.46


7. 训练次数
baseline: FPN结构，avg先conv-bn-sig再乘到FPN上面，先interpolate再1x1conv，不加auxiliary
screen: 13207
* 11000次: 
* 31000次: 
* 51000次: 



7. 使用90k个iter去训练
* cp变成FPN(bilinear), sp 64, 256, 512:
54.20 - 提高个点
* cp变成FPN(nearest), sp 64, 256, 512: 晚
* cp先FPN再channel-atten相乘(avg做conv-bn-sigmoid再乘): 更晚

* 先interpolate到原图大小，再conv1x1


* 对arm输出先conv-bn-relu再interpolate，再1x1conv再用auxliary_loss: 


* interpolate之后再加上conv-bn-relu


7. 使用自己的训练sheduel，两个step这种



2. eval 时像论文那样使用1024x1024

2. 如果不收敛就先warmup一下再弄
