from typing import Any
import ailang as al
import ailang.nn as nn



#[TODO] 继承nn.Module的时候会有问题 [DONE]

class myConv(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size)
        self.conv2 = nn.Conv2d(out_channels,out_channels,5)
        # x = al.tensor((1,3,224,224), "Float") 
        # print(type(c1(x)))
        # print(al.compile_ir(c1(x),x))

    def __call__(self,x):
        ir = self.conv1(x)
        ir2 = self.conv2(x)
        print(al.compile_ir(ir,x))
        print(al.compile_ir(ir2,x))
    
model_ = myConv(224,224,3)



print(model_.parameters())
x = al.tensor((1,3,224,224), "Float") 
model_(x)




