import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from tool_KL import KL_divergence
from Attention import se_block,cbam_block,eca_block 
from tool import Spatial_Dropout
__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']
attention_block = [se_block,cbam_block,eca_block]


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        
        self.features_deep, num_features = self.make_features_demo(growth_rate, block_config,
                 num_init_features, bn_size, drop_rate,in_channel=3) 
        self.features_hand1, num_features = self.make_features_demo(growth_rate, block_config,
                 num_init_features, bn_size, drop_rate,in_channel=1)  


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        
        # classifier
      
        self.classifier_deep100 = self.make_classifier_demo(num_features,100)
        self.classifier_hand1 = self.make_classifier_demo(num_features,100)

        #HOG:23328  #LBP:26 #Harris: 512 #Sift 46208  # encoder 2048 
        #LBP_ori:50176  # Harris_ori:50176    # Gabor:50176  # Canny：50176
        
        self.fc_0 = self.make_fc_demo(100,num_classes)
        self.fc_1 = self.make_fc_demo(100,num_classes)
      
        self.fc_final = self.make_fc_demo(2048,num_classes)

        self.fusion_11 = nn.Linear(200, 100)
        self.fusion_21 = nn.Linear(200, 100)


        self.attention = attention_block[2](num_features*2) # attention
        self.sdropput_1 = Spatial_Dropout(0.5)

        
    
    def make_features_demo(self,growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,in_channel=3):
        
        features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channel, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        return features,num_features

    def make_classifier_demo(self,num_features,out_size):
        classifier = nn.Sequential(
            # nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(num_features, out_size)
        )

        return classifier

    def make_fc_demo(self,in_size, out_size):
        classifier = nn.Sequential(
            # nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(in_size, out_size)
        )
        return classifier




    #'''MFCFNet + 2D hand + syloss
    def forward(self,x,F2D_1,label):
        criterion = nn.CrossEntropyLoss()

        x1 = self.features_deep(x)
        x1 = F.relu(x1, inplace=True)
        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        map_deep = x1
        x1 = torch.flatten(x1, start_dim=1)
        x1_f = self.classifier_deep100(x1)

        x1_fc0 = self.fc_0(x1_f)
        L0 = torch.mean(criterion(x1_fc0, label))
        s_deep = torch.softmax(x1_fc0,dim=0)

        F2D_1 = F2D_1.view([-1,1,128,128]) 
        D1 = self.features_hand1(F2D_1)
        D1 = F.relu(D1, inplace=True)
        D1 = F.adaptive_avg_pool2d(D1, (1, 1))
        map_hand = D1
        D1 = torch.flatten(D1, start_dim=1)
        D1_f = self.classifier_hand1(D1)

        D1_fc1 = self.fc_1(D1_f)
        L1 = torch.mean(criterion(D1_fc1, label))
        s_hand= torch.softmax(D1_fc1,dim=0)

        concat1 = torch.cat((x1_f, D1_f), dim=1)
        
        # Fusion
        map_cat = torch.cat((map_deep, map_hand), dim=1)
       

        fusion = self.sdropput_1(map_cat)
        fusion = self.attention(fusion) 
        fusion = torch.flatten(fusion, start_dim=1)

        f11 = self.fusion_11(concat1)
        f21 = self.fusion_21(concat1)


        alpha1 = torch.sigmoid(f11)
        alpha2 = torch.sigmoid(f21)
        f1 = torch.multiply(alpha1, x1_f)
        f2 = torch.multiply(alpha2, D1_f)

        
        f14 = torch.sum(f1)
        f24 = torch.sum(f2)

        f4 = torch.stack([f14, f24])

        beta = torch.softmax(f4, dim=0)
        f13 = beta[0]
        f23 = beta[1]

        L1_2 = KL_divergence(f1,f2)  
        L2_1 = KL_divergence(f2,f1)
        

        ######################### Fusion ################################

        x = torch.cat((f1, f2), dim=1)  
        x = self.fc_final(fusion)  


        return x, f13*L0 + f23*L1 + 0.5*(L1_2+L2_1) 
    # ''' 

 

def densenet121( **kwargs):

    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model

def densenet169(**kwargs):
   
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    return model

def densenet201(**kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
   
    return model