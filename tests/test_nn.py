import ailang as al
import ailang.nn as nn

class TestNN:
    def test_conv_once(self):
        class myConv(nn.Module):
            def __init__(self,in_channels,out_channels,kernel_size):
                super().__init__()
                self.kernel_size = kernel_size
                self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size)
                self.conv2 = nn.Conv2d(out_channels,out_channels,5)
                
            def __call__(self,x):
                ir = self.conv1(x)
                ir2 = self.conv2(x)
                print(al.compile_ir(ir,x))
                print(al.compile_ir(ir2,x))
            
        model_ = myConv(224,224,3)