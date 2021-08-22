from lib import *

class L2Norm(nn.Module):
    def __init__(self,input_channels=512, scale=20):
        super(L2Norm,self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_channels))
        self.scale=scale
        self.reset_parameters()
        self.eps=1e-10
    
    def reset_parameters(self):
        nn.init.constant_[self.weights, self.scale]
    
    def forward(self,x):
        # x.size()= (batch_size, channels, height,width)
        norm = x.pow(2).sum(dim=1, keepdims=True).sqrt() * self.eps
        x= torch.div(x,norm)
        # 512 --> 1,512,1,1
        weights = self.weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return weights*x