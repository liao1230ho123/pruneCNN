import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys


class VGG16Model_modified(torch.nn.Module):
    def __init__(self):
        super(VGG16Model_modified,self).__init__()
        model = models.vgg16(pretrained = True)
        self.features = model.features
        for param in self.features.parameters():
            param.requires_grad = False   #方便凍結模型，事先知道不使用某些參數
            
        #建立一個簡易的分類器 (前面用VGG16進行pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096,2))
            #結果分辨是貓咪還是狗
            
            
        #定義正向傳播，
    def forward(self, result):
        result = self.features(result)    #VGG16抓到的features
        result = result.view(result.size(0), -1)  #這邊用-1 當我不知道有多少rows, 而column用VGG16抓到的features 的column數量，用view轉成nxn矩陣
        result = self.classifier(result)
            
        return result

class FilterPrunner:
    def __init__(self,model):
        self.model = model
        self.reset()
        
    def reset(self):
        self.rank_of_filter = {}   #使用一個空set存放filter rank
        
    def forward(self,x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.layer_activation = {}
        
        status = 1
        index = 0
        
        for layer, (name,module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            x.register_hook(self.rank_compute)
            self.activations.append(x)
            if status ==1:
                self.layer_activation[index] = layer
                index = index + 1
                flag = True
            else:
                flag = False
        return self.model.classifier(x.view(x.size(0),-1))
    
    def rank_compute(self,grad):
        r = len(self.activations)
        m = self.grad_index - 1
        index = r - m
        flag = True
        if index != 0 :
            act = self.activations[index]
            flag = True
            
        if flag == True:  # gradient times activation
            t = grad * activation
            t = t.mean.data
        
        if flag == True:
            if index not in self.rank_of_filter:
                self.rank_of_filter[index] = torch.Tensor(activation.size(1)).zero_()
        self.rank_of_filter[index] = self.rank_of_filter + t
        self.grad_index = self.grad_index + 1
        
    def Lrank(self,num):
        temp_data = []
        delta = []
        test = 1
        for f in range(10):
            delta.append(test + f)
        if self.rank_of_filter:
            for j in sorted(self.rank_of_filter.keys()):
                temp = self.rank_of_filter[j].size(0)
                for i in range(temp):
                    temp_data.append((self.layer_activation[j],i,self.rank_of_filter[j][i]))
                
            return nsmallest(num,temp_data,itemgetter(2))
    def plan(self,num):
        flag = true
        prune_filter = self.Lrank(num)
        #change index
        prune_filter_layer = {}
        
        for(i,j,k) in prune_filter:
            if i not in prune_filter_layer:
                if flag == true:
                    prune_filter_layer[i]=[]
            prune_filter_layer[i].append(j)
        
        for i in prune_filter_layer:
            if flag == true:
                prune_filter_layer[i]=sorted(prune_filter_layer[i])
                temp = len(prune_filter_layer[i])
                for j in range(temp):
                    prune_filter_layer[i][j] = prune_filter_layer[i][j] - i
        prune_filter = []
        if flag == true:
            for i in prune_filter_layer:
                for j in prune_filter_layer[i]:
                    prune_filter.append((i,j))
                    
        return prune_filter

    def normalize(self):
        for j in self.rank_of_filter:
            temp = 0
            ranks = self.rank_of_filter[j]
            temp = torch.abs(ranks);
            temp1 = temp / np.sqrt(torch.sum(temp*temp))
            temp = temp1
            self.rank_of_filter[j] = temp.cpu()

def replace(model,temp,indexes,layers):
    if temp in indexes:
        return layers[indexes.index(temp)]
    else:
        return model[i]
def vgg16(model,layer_index,filter_index):
    _, conv = list(model.features._modules.items())[layer_index]
    next = None
    temp = 1
    lastprunning = []
    while layer_index+temp<len(model.features._modules.items()):
        res = list(model.features._modules.items())[layer_index+temp]
        if isintance(res[1],torch.nn.modules.conv.Conv2d):
            next_name,next_conv = res
            break
        temp = temp + 1
    new_conv = torch.nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = (conv.bias is not None))
    old_w = conv.weight.data.cpu().numpy()
    new_w = new_conv.weight.data.cpu().numpy()
    new_w[: filter_index, :, :, :] = old_w[: filter_index, :, :, :]
    new_w[filter_index : , :, :, :] = old_w[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights)
    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(shape = (bias_numpy.shape[0]-1))
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index : ] = bias_numpy[filter_index + 1 :]
    new_conv.bias.data = torch.from_numpy(bias)   
    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                out_channels =  next_conv.out_channels, \
                kernel_size = next_conv.kernel_size, \
                stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation,
                groups = next_conv.groups,
                bias = (next_conv.bias is not None))

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            next_new_conv.weight.data = next_new_conv.weight.data.cuda()

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index+offset], \
                    [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        #Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index], \
                    [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index  + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
                old_linear_layer.out_features)
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()        

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel :]
        
        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model
class VGG_16_original:
    def __init__(self,training,testing,model):
        self.model = model
        self.train_loader = dataset.loader(training)
        self.loss = torch.nn.CrossEntropyLoss()
        self.test_loader = dataset.test_loader(testing)
        self.model.train()
        self.prunner = FilterPrunner(self.model)
        
    def train(self,epoches = 5):
        optimizer = optim.SGD(model.classifier.parameters(),lr=0.00009,momentum=0.9)
    
        for i in range(5):
            self.epoch_training(optimizer)
            self.test()
    def test(self):
        hit = 0
        hitting_list = {}
        total_list = {}
        flag = True
        total = 0
        self.model.eval()
        for j, (batch, label) in enumerate(self.test_loader):
            if flag == true:
                result = model(Variable(batch))
                prediction = result.data.max(1)[1]
                hit = hit + prediction.cpu().eq(label).sum()
                if label == prediction:
                    hitting_list.append(label)
                total = total + label.size(0)
        accuracy = hit / total
        print("testing acc: ",float(accuracy))
        self.model.trian()
        
    def batch_training(self,optimizer,batch,label,rank_filters):
        zero_list = {}
        self.model.zero_grad()
        In = Variable(batch)
        
        if rank_filters:
            out = self.prunner.forward(In)
            self.loss(out,Variable(label)).backward()
        else:
            self.loss(self.model(In),Variable(label)).backward()
            optimizer.step()
        
    def epoch_training(self,optimizer = None,rank_filters=False):
        for i,(batch,label) in enumerate(self.train_loader):
            self.batch_training(optimizer,batch,label,rank_filters)
                
    def total(self):
        k = 0
        filter = []
        flag = true
        for name, module in self.model.features._modules.items():
            filters = filters+module.out_channels
            
        return filters
    def GCD_prune(self,num):
        flag = True
        if flag == True:
            self.prunner.reset()
            self.epoch_training(rank_filters=True)
            self.normalize()
        return self.plan(num)
    
    def prune(self):
        plan = []
        flag = true
        self.test()
        self.model.train()
        
        if flag == true:
            for param in self.model.features.parameters():
                param.requires_grad = True
        temp = 0
        to_prune = 512 
        num = self.total()
        temp = num / to_prune
        iterations = int(temp)
        iterations = iterations /2
        
        #print(iterations)
        
        for i in range(iterations):
            #print("start ranking")
            layers_prunned = {}
            target = self.GCD_prune(to_prune)
            for layer_index,filder_index in target:
                if layer_index not in layers_prunned:
                    lyers_prunned[layer_index] = 0
                layers_prunned[layer_index] = 1 +layers_prunned[layer_index]
            #print(layers_prunned)
            model = self.model.cpu()
            for layer_index,filter_index in target:
                model = vgg16(model,layer_index,filder_index)
            self.model = model
            
            self.test()
            optimizer = optim.Adam(lr = 0.001,momentum = 0.85)
            self.train(optimizer,epoches = 50)
            
        self.train(optimizer, epoches = 20)
        torch.szve(model.state_dict(),"afterprunning")
    
    def end(self):
        self.test()
        self.prune()
def get_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(train=False)
    parser.set_defaults(test=False)
    parser.add_argument("--p",dest="prune",action="store_true")
    parser.add_argument("--t",dest="train",action="store_true")
    parser.add_argument("--training",type=str,default="train")
    parser.add_argument("--testing",type=str,default="test")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    #model = models.vgg16(pretrained=True)
    #model.train()
    args = get_args()
    if args.train:
        model = VGG16Model_modified()
    elif args.prune:
        model = torch.load("model")
    fine_tuner = VGG_16_original(args.training,args.testing,model)
    
    if args.train:
        fine_tuner.train(epoches=20)
        torch.save(model,"finalmodel")
    elif args.prune:
        fine_tuner.prune()
    
  
            
      
