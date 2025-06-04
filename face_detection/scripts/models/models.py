import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class L2Norm(nn.Module) : 

    def __init__(self , n_channels : int , scale = 1.0) -> None : 

        super(L2Norm , self).__init__()

        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def forward(self , inp : Tensor) -> Tensor : 

        norm = inp.pow(2).sum(dim = 1 , keepdim = True).sqrt() + self.eps

        inp = inp / norm * self.weight.view(1 , -1 , 1 , 1)

        return inp
    
class Conv1(nn.Module) : 
    
    def __init__(self) -> None : 
        
        super(Conv1 , self).__init__()
        
        self.conv1 = nn.Conv2d(3 , 64 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv2 = nn.Conv2d(64 , 64 , kernel_size = 3 , stride = 1 , padding = 1)
        
    def forward(self , inp : Tensor) -> Tensor : 
        
        conv1_o = F.relu(self.conv1(inp))
        conv2_o = F.relu(self.conv2(conv1_o))
        
        output = F.max_pool2d(conv2_o , 2 , 2)
        
        return output
class Conv2(nn.Module) : 
    
    def __init__(self) -> None : 
        
        super(Conv2 , self).__init__()
        
        self.conv1 = nn.Conv2d(64 , 128 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv2 = nn.Conv2d(128 , 128 , kernel_size = 3 , stride = 1 , padding = 1)
        
    def forward(self , inp : Tensor) -> Tensor : 
        
        conv1_o = F.relu(self.conv1(inp))
        conv2_o = F.relu(self.conv2(conv1_o))
        
        output = F.max_pool2d(conv2_o , 2 , 2)
        
        return output

class Conv3(nn.Module) : 
    
    def __init__(self) -> None : 
        
        super(Conv3 , self).__init__()
        
        self.conv1 = nn.Conv2d(128 , 256 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv2 = nn.Conv2d(256 , 256 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv3 = nn.Conv2d(256 , 256 , kernel_size = 3 , stride = 1 , padding = 1)

    def forward(self , inp : Tensor) -> Tensor : 
        
        conv1_o = F.relu(self.conv1(inp))
        conv2_o = F.relu(self.conv2(conv1_o))
        conv3_o = F.relu(self.conv3(conv2_o))
        
        return conv3_o
class Conv4(nn.Module) : 
    
    def __init__(self) -> None : 
        
        super(Conv4 , self).__init__()
        
        self.conv1 = nn.Conv2d(256 , 512 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv2 = nn.Conv2d(512 , 512 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv3 = nn.Conv2d(512 , 512 , kernel_size = 3 , stride = 1 , padding = 1)

    def forward(self , inp : Tensor) -> Tensor : 
        
        conv1_o = F.relu(self.conv1(inp))
        conv2_o = F.relu(self.conv2(conv1_o))
        conv3_o = F.relu(self.conv3(conv2_o))
        
        return conv3_o

class Conv5(nn.Module) : 
    
    def __init__(self) -> None : 
        
        super(Conv5 , self).__init__()
        
        self.conv1 = nn.Conv2d(512 , 512 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv2 = nn.Conv2d(512 , 512 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv3 = nn.Conv2d(512 , 512 , kernel_size = 3 , stride = 1 , padding = 1)

    def forward(self , inp : Tensor) -> Tensor : 
        
        conv1_o = F.relu(self.conv1(inp))
        conv2_o = F.relu(self.conv2(conv1_o))
        conv3_o = F.relu(self.conv3(conv2_o))
        
        return conv3_o

class Conv6(nn.Module) : 
    
    def __init__(self) -> None : 
        
        super(Conv6 , self).__init__()
        
        self.conv1 = nn.Conv2d(512 , 1024 , kernel_size = 3 , stride = 1 , padding = 3)
        self.conv2 = nn.Conv2d(1024 , 1024 , kernel_size = 1 , stride = 1 , padding = 0)
        
    def forward(self , inp : Tensor) -> Tensor : 
        
        conv1_o = F.relu(self.conv1(inp))
        conv2_o = F.relu(self.conv2(conv1_o))
        
        return conv2_o
class Conv7(nn.Module) : 
    
    def __init__(self) -> None : 
        
        super(Conv7 , self).__init__()
        
        self.conv1 = nn.Conv2d(1024 , 256 , kernel_size = 1 , stride = 1 , padding = 0)
        self.conv2 = nn.Conv2d(256 , 512 , kernel_size = 3 , stride = 2 , padding = 1)
        
    def forward(self , inp : Tensor) -> Tensor : 
        
        conv1_o = F.relu(self.conv1(inp))
        conv2_o = F.relu(self.conv2(conv1_o))
        
        return conv2_o
class Conv8(nn.Module) : 
    
    def __init__(self) -> None : 
        
        super(Conv8 , self).__init__()
        
        self.conv1 = nn.Conv2d(512 , 128 , kernel_size = 1 , stride = 1 , padding = 0)
        self.conv2 = nn.Conv2d(128 , 256 , kernel_size = 3 , stride = 2 , padding = 1)
        
    def forward(self , inp : Tensor) -> Tensor : 
        
        conv1_o = F.relu(self.conv1(inp))
        conv2_o = F.relu(self.conv2(conv1_o))
        
        return conv2_o

class s3fd(nn.Module) : 

    def __init__(self) : 

        super(s3fd, self).__init__()

        self.conv1 = Conv1()
        self.conv2 = Conv2()

        self.conv3 = Conv3()
        self.conv4 = Conv4()
        
        self.conv5 = Conv5()
        
        self.conv6 = Conv6()
        self.conv7 = Conv7()
        self.conv8 = Conv8()

        self.conv3_3_norm = L2Norm(256 , scale = 10)
        self.conv4_3_norm = L2Norm(512 , scale = 8)
        self.conv5_3_norm = L2Norm(512 , scale = 5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256 , 4 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256 , 4 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512 , 2 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512 , 4 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv5_3_norm_mbox_conf = nn.Conv2d(512 , 2 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512 , 4 , kernel_size = 3 , stride = 1 , padding = 1)

        self.fc7_mbox_conf = nn.Conv2d(1024 , 2 , kernel_size = 3 , stride = 1 , padding = 1)
        self.fc7_mbox_loc = nn.Conv2d(1024 , 4 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv6_2_mbox_conf = nn.Conv2d(512 , 2 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv6_2_mbox_loc = nn.Conv2d(512 , 4 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv7_2_mbox_conf = nn.Conv2d(256 , 2 , kernel_size = 3 , stride = 1 , padding = 1)
        self.conv7_2_mbox_loc = nn.Conv2d(256 , 4 , kernel_size = 3 , stride = 1 , padding = 1)

    def forward(self, x) : 

        h = self.conv1(x) ; h = self.conv2(h)

        # ! -------------------------------------------CLS 1-------------------------------------------

        h = self.conv3(h) ; f3_3 = self.conv3_3_norm(h)
        cls1 = self.conv3_3_norm_mbox_conf(f3_3) ; reg1 = self.conv3_3_norm_mbox_loc(f3_3)

        chunk = torch.chunk(cls1 , 4 , 1)
        bmax = torch.max(torch.max(chunk[0] , chunk[1]) , chunk[2])
        cls1 = torch.cat([bmax , chunk[3]] , dim = 1)

        yield cls1 , reg1
        

        # ! -------------------------------------------CLS 2-------------------------------------------

        h = F.max_pool2d(h , 2 , 2)
        h = self.conv4(h) ; f4_3 = self.conv4_3_norm(h)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3) ; reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        
        yield cls1 , reg1
        

        # ! -------------------------------------------CLS 3-------------------------------------------

        h = F.max_pool2d(h , 2 , 2)
        h = self.conv5(h) ; f5_3 = self.conv5_3_norm(h)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3) ; reg3 = self.conv5_3_norm_mbox_loc(f5_3)

        yield cls3 , reg3        


        # ! -------------------------------------------CLS 4-------------------------------------------

        
        h = F.max_pool2d(h , 2 , 2)
        h = self.conv6(h) ; ffc7 = h
        cls4 = self.fc7_mbox_conf(ffc7) ; reg4 = self.fc7_mbox_loc(ffc7)

        yield cls4 , reg4
        

        # ! -------------------------------------------CLS 5-------------------------------------------

        h = self.conv7(h) ; f6_2 = h
        cls5 = self.conv6_2_mbox_conf(f6_2) ; reg5 = self.conv6_2_mbox_loc(f6_2)

        yield cls5 , reg5


        # ! -------------------------------------------CLS 6-------------------------------------------

        h = self.conv8(h) ; f7_2 = h
        cls6 = self.conv7_2_mbox_conf(f7_2) ; reg6 = self.conv7_2_mbox_loc(f7_2)

        yield cls6 , reg6