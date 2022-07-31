import torch.nn as nn
import torch

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps, momentum):
        super(BatchNorm, self).__init__()
        self.batchnorm = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)
        self.unbatchnorm = nn.MaxUnpool2d(num_features=num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        self.X = x

        output, self.indices = self.batchnorm(x)

        return output

    def learn_pattern(self, x):
        return self.forward(x)

    def compute_pattern(self):
        pass

    # def gradient_backward(self, R):
    #     return self.lrp_backward(R)
    #
    # def lrp_backward(self, R):
    #
    #     batch_size, channels, height, width = self.X.shape
    #     height = int(height/2)
    #     width = int(width/2)
    #
    #     if R.shape != torch.Size([batch_size, channels, height, width]):
    #         R = R.view(batch_size, channels, height, width)
    #
    #     return self.unpool(R, self.indices)
    #
    # def analyze(self, method, R):
    #
    #     batch_size, channels, height, width = self.X.shape
    #     height = int(height/2)
    #     width = int(width/2)
    #
    #     if R.shape != torch.Size([batch_size, channels, height, width]):
    #         R = R.view(batch_size, channels, height, width)
    #
    #     return self.unpool(R, self.indices)