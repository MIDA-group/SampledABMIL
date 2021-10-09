""" Attention-based Deep Multiple Instance Learning (ABMIL) models based on Ilse et el. paper [1] for experiments with one and three GPUs, for QMNIST and Imagenette datasets. "Attention1GPU" corresponds to original "Attenion" model that can be found in https://github.com/AMLab-Amsterdam/AttentionDeepMIL. 
[1] Ilse, Maximilian, Jakub Tomczak, and Max Welling. "Attention-based deep multiple instance learning." International conference on machine learning. PMLR, 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck


class Attention3GPUs(nn.Module):
    def __init__(self):
        super(Attention3GPUs, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1_0 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
        ).to('cuda:2')
    
        self.feature_extractor_part1_1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
        ).to('cuda:0')
        
        self.feature_extractor_part1_2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        ).to('cuda:1') 
        
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        ).to('cuda:1')

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        ).to('cuda:1')

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        ).to('cuda:1')

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1_0(x.to('cuda:2'))
        H = self.feature_extractor_part1_1(H.to('cuda:0'))
        H = self.feature_extractor_part1_2(H.to('cuda:1'))
        
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H).to('cuda:1')  # NxL

        A = self.attention(H).to('cuda:1')  # NxK
        A = torch.transpose(A.to('cuda:1'), 1, 0)  # KxN
#         print('A.shape', A.shape)
        A = F.softmax(A.to('cuda:1'), dim=1)  # softmax over N
#         print('A.shape,soft', A.shape)

        M = torch.mm(A.to('cuda:1'), H)  # KxL

        Y_prob = self.classifier(M.to('cuda:1'))
        Y_hat = torch.ge(Y_prob.to('cuda:1'), 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()#.data[0]
#         print('error:', error, 'Y_hat', Y_hat)
        
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class Attention1GPU(nn.Module):
    def __init__(self):
        super(Attention1GPU, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1_0 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
        )
    
        self.feature_extractor_part1_1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
        )
        
        self.feature_extractor_part1_2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        ) 
        
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1_0(x)
        H = self.feature_extractor_part1_1(H)
        H = self.feature_extractor_part1_2(H)
        
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H) # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()#.data[0]
#         print('error:', error, 'Y_hat', Y_hat)
        
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    

num_classes = 2
class ModelParallelResNet18(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet18, self).__init__(
            Bottleneck, [2, 2, 2, 2], num_classes=num_classes, *args, **kwargs)        
        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1
        ).to('cuda:2')
        
        self.seq2 = nn.Sequential(
            self.layer2
        ).to('cuda:0')

        self.seq3 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool
        ).to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:0'))
        x = self.seq3(x.to('cuda:1'))
        return x.view(x.size(0), -1) 
    
    
class Attention_Imagenette_bags_3GPUs(nn.Module):
    def __init__(self):
        super(Attention_Imagenette_bags_3GPUs, self).__init__()
        self.L = 2048 
        self.D = 524  
        self.K = 1

        self.feature_extractor = ModelParallelResNet18()      

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        ).to('cuda:1')

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        ).to('cuda:1')

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x).to('cuda:1')
        A = self.attention(H.to('cuda:1'))  # NxK
        A = torch.transpose(A.to('cuda:1'), 1, 0)  # KxN
        A = F.softmax(A.to('cuda:1'), dim=1)  # softmax over N

        M = torch.mm(A.to('cuda:1'), H)  # KxL

        Y_prob = self.classifier(M.to('cuda:1'))
        Y_hat = torch.ge(Y_prob.to('cuda:1'), 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()#.data[0]
        
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class ModelParallelResNet18_1GPU(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet18_1GPU, self).__init__(
            Bottleneck, [2, 2, 2, 2], num_classes=num_classes, *args, **kwargs)        
        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1
        )
        
        self.seq2 = nn.Sequential(
            self.layer2
        )

        self.seq3 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool
        )

    def forward(self, x):
        x = self.seq2(self.seq1(x))
        x = self.seq3(x)
        return x.view(x.size(0), -1) 
    
class Attention_Imagenette_bags_1GPU(nn.Module):
    def __init__(self):
        super(Attention_Imagenette_bags_1GPU, self).__init__()
        self.L = 2048 
        self.D = 524  
        self.K = 1

        self.feature_extractor = ModelParallelResNet18_1GPU()      

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x)
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()#.data[0]
        
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A