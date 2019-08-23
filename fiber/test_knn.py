#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 3)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 20)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 3)
parser.add_argument("-e","--episode",type = int, default= 30)
parser.add_argument("-t","--test_episode", type = int, default = 500)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
device = torch.device("cuda:"+str(GPU) if torch.cuda.is_available() else "cpu")


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*12*12,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatest_folders = tg.fiber_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)


    feature_encoder.to(device)
    relation_network.to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    model_way = 4
    model_shot = 5
    if os.path.exists(str("./models/fiber_feature_encoder_" + str(model_way) +"way_" + str(model_shot) +"shot.pt")):
        feature_encoder.load_state_dict(torch.load(str("./models/fiber_feature_encoder_" + str(model_way) +"way_" + str(model_shot) +"shot.pt"), map_location=device))
        print("load feature encoder success")
    if os.path.exists(str("./models/fiber_relation_network_"+ str(model_way) +"way_" + str(model_shot) +"shot.pt")):
        relation_network.load_state_dict(torch.load(str("./models/fiber_relation_network_"+ str(model_way) +"way_" + str(model_shot) +"shot.pt"), map_location=device))
        print("load relation network success")

    # feature_encoder.eval()
    # relation_network.eval()

    total_accuracy = 0.0
    for episode in range(EPISODE):
            torch.cuda.empty_cache()
            
            # test
            print("Testing...")

            accuracies = []
            for i in range(TEST_EPISODE):
                torch.cuda.empty_cache()
                total_rewards = 0

                task = tg.FiberTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                sample_dataloader = tg.get_fiber_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                test_dataloader = tg.get_fiber_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=False)

                sample_images,sample_labels = sample_dataloader.__iter__().next()
                
                for test_images,test_labels in test_dataloader:
                    torch.cuda.empty_cache()
                    batch_size = test_labels.shape[0]
                    
                    with torch.no_grad():
                        # calculate features
                        sample_features = feature_encoder(Variable(sample_images).to(device)) # [20*3,64,54,54]
                        
                        # make relation pair each sample
                        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1) # [10*3,20*3,64,54,54]

                        # sum up all the samples
                        # sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,54,54) # [3,20,64,54,54]
                        # sample_features = torch.sum(sample_features,1).squeeze(1) # [3,64,54,54]
                        #sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1) # [30,3,64,54,54]

                        test_features = feature_encoder(Variable(test_images).to(device)) # [10*3,64,54,54]
                        test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) # [20*3,10*3,64,54,54]
                        # test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1) # [3,30,64,54,54]
                        test_features_ext = torch.transpose(test_features_ext,0,1) # [10*3,20*3,64,54,54]

                        # calculate relations
                        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,54,54) # [30*60,128,54,54]
                        relations = relation_network(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS) # [30,60]
                        # print(relations.data)

                        # predict the labels
                        # _,predict_labels = torch.max(relations.data,1) # the max value as predict label
                        top_num = 20
                        _,relation_score = torch.topk(relations.data, top_num, dim=1, largest=True) # return top 20 values
                        relation_score = relation_score/SAMPLE_NUM_PER_CLASS
                        # print("relation score : ", relation_score)
                        
                        # count the number of each class
                        predict_labels = torch.zeros(CLASS_NUM*BATCH_NUM_PER_CLASS)
                        for n in range(CLASS_NUM*BATCH_NUM_PER_CLASS):
                            predict = torch.zeros(CLASS_NUM,1)
                            for k in range(top_num):
                                if relation_score[n][k] == 0:
                                    predict[0] += 1
                                elif relation_score[n][k] == 1:
                                    predict[1] += 1
                                else:
                                    predict[2] += 1
                            _,predict_label = torch.max(predict.data,0)
                            predict_labels[n] = predict_label

                        predict_labels = predict_labels.to(device,dtype=torch.int32)
                        test_labels = test_labels.to(device,dtype=torch.int32)
                        
                        # print("predict labels: ", predict_labels)
                        # print("test labels:    ", test_labels)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)


                accuracy = total_rewards/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)

            total_accuracy += test_accuracy

    print("aver_accuracy:",total_accuracy/EPISODE)






if __name__ == '__main__':
    main()
