import torch
import cppcuda_tutorial   # 导入自己写的扩展库，定义在setpup.py中
import time
N=10
F=10
feats=torch.randn(N,4,F).cuda()
points=torch.randn(N,2).cuda()
#双线性插值函数通过c++实现
out=cppcuda_tutorial.f_b(feats,points)
print(feats)
print('-------------')
print(points)
print('-------------')
print(out)




