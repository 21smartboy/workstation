import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import MAML.label_classify as cla  #将以某个字符开头的所有病毒给一个统一的名称
import MAML.find_variety as fv   #将特定种类的病毒选出来

data=pd.read_csv('E:/Dataset/CIC-IOT2013/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',skiprows=1)
x=data.iloc[:,:-1]  #取出最后一列外的所有
y=data.iloc[:,-1]  #取出最后一列
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=42)

ddoS=fv.find_variety(y_train,"DDoS") #找出以DDoS开头的病毒
target='Attack'
y_lab=cla.classify(target,ddoS,y_train)
#2  33006
doS=fv.find_variety(y_lab,'DoS')
target='Attack'
y_lab=cla.classify(target,doS,y_lab)
#3  10789
mirai=fv.find_variety(y_lab,'Mirai')
target='Attack'
y_lab=cla.classify(target,mirai,y_lab)
#4  4528
benign=fv.find_variety(y_lab,'Benign')
target='Benign'
y_lab=cla.classify(target,benign,y_lab)
#5  1314
mITM=fv.find_variety(y_lab,'MITM')
target='Attack'
y_lab=cla.classify(target,mITM,y_lab)
#6  1324
recon=fv.find_variety(y_lab,'Recon')
target='Attack'
y_lab=cla.classify(target,recon,y_lab)
#7  22
browserHijacking=fv.find_variety(y_lab,'Bro')
target='Attack'
y_lab=cla.classify(target,browserHijacking,y_lab)
#8  53
dictionaryBruteForce=fv.find_variety(y_lab,'Dictionary')
target='Attack'
y_lab=cla.classify(target,dictionaryBruteForce,y_lab)
#9  738
dNS_Spoofing=fv.find_variety(y_lab,'DNS')
target='Attack'
y_lab=cla.classify(target,dNS_Spoofing,y_lab)
#10  153
vulnerabilityScan=fv.find_variety(y_lab,'Vulner')
target='Attack'
y_lab=cla.classify(target,vulnerabilityScan,y_lab)
#11  25
sqlInjection=fv.find_variety(y_lab,'Sql')
target='Attack'
y_lab=cla.classify(target,sqlInjection,y_lab)
#12 15
xSS=fv.find_variety(y_lab,'XSS')
target='Attack'
y_lab=cla.classify(target,xSS,y_lab)
#13 20
commandInjection=fv.find_variety(y_lab,'CommandInjection')
target='Attack'
y_lab=cla.classify(target,commandInjection,y_lab)
#14 6
uploading_Attack=fv.find_variety(y_lab,'Uploading_Attack')
target='Attack'
y_lab=cla.classify(target,uploading_Attack,y_lab)
#15  18
backdoor=fv.find_variety(y_lab,'Backdoor')
target='Attack'
y_trainlab=cla.classify(target,backdoor,y_lab)
# print('y_label结果:',y_lab)

#以下为测试集
ddoS=fv.find_variety(y_test,"DDoS") #找出以DDoS开头的病毒
target='Attack'
y_lab=cla.classify(target,ddoS,y_train)
#2
doS=fv.find_variety(y_lab,'DoS')
target='Attack'
y_lab=cla.classify(target,doS,y_lab)
#3
mirai=fv.find_variety(y_lab,'Mirai')
target='Attack'
y_lab=cla.classify(target,mirai,y_lab)
#4
benign=fv.find_variety(y_lab,'Benign')
target='Benign'
y_lab=cla.classify(target,benign,y_lab)
#5
mITM=fv.find_variety(y_lab,'MITM')
target='Attack'
y_lab=cla.classify(target,mITM,y_lab)
#6
recon=fv.find_variety(y_lab,'Recon')
target='Attack'
y_lab=cla.classify(target,recon,y_lab)
#7
browserHijacking=fv.find_variety(y_lab,'Bro')
target='Attack'
y_lab=cla.classify(target,browserHijacking,y_lab)
#8
dictionaryBruteForce=fv.find_variety(y_lab,'Dictionary')
target='Attack'
y_lab=cla.classify(target,dictionaryBruteForce,y_lab)
#9
dNS_Spoofing=fv.find_variety(y_lab,'DNS')
target='Attack'
y_lab=cla.classify(target,dNS_Spoofing,y_lab)
#10
vulnerabilityScan=fv.find_variety(y_lab,'Vulner')
target='Attack'
y_lab=cla.classify(target,vulnerabilityScan,y_lab)
#11
sqlInjection=fv.find_variety(y_lab,'Sql')
target='Attack'
y_lab=cla.classify(target,sqlInjection,y_lab)
#12
xSS=fv.find_variety(y_lab,'XSS')
target='Attack'
y_lab=cla.classify(target,xSS,y_lab)
#13
commandInjection=fv.find_variety(y_lab,'CommandInjection')
target='Attack'
y_lab=cla.classify(target,commandInjection,y_lab)
#14
uploading_Attack=fv.find_variety(y_lab,'Uploading_Attack')
target='Attack'
y_lab=cla.classify(target,uploading_Attack,y_lab)
#15
backdoor=fv.find_variety(y_lab,'Backdoor')
target='Attack'
y_testlab=cla.classify(target,backdoor,y_lab)


class part1_data(Dataset):
    def __init__(self,data,label):
        scaler=MinMaxScaler()    #归一化
        self.predata=scaler.fit_transform(data)#进行归一化返回numpy.ndarray
        self.predata=torch.from_numpy(self.predata)  #将numpy.ndarray类型转化成张量
        self.new_label = LabelEncoder().fit_transform(label)  #对标签进行编码
        self.label=torch.tensor(self.new_label)

    def __getitem__(self, index):
        return self.predata[index], self.label[index]

    def __len__(self):
        return len(self.predata)

dataset1=part1_data(x_train,y_trainlab)
train_loader=DataLoader(dataset1,shuffle=True,batch_size=128)
dataset2=part1_data(x_test,y_testlab)
test_loader=DataLoader(dataset2,shuffle=False,batch_size=128)

class my_modle(nn.Module):
    def __init__(self):
        super(my_modle,self).__init__()
        self.leaner1=nn.Linear(46,32)
        self.leaner2=nn.Linear(32,16)
        self.leaner3=nn.Linear(16, 8)
        self.leaner4 = nn.Linear(8, 4)
        self.leaner5 = nn.Linear(4, 2)
    def forward(self,x):
        x=x.to(torch.float)
        x=x.view(-1,46)
        x=F.relu(self.leaner1(x))
        x=F.relu(self.leaner2(x))
        x=F.relu(self.leaner3(x))
        x=F.relu(self.leaner4(x))
        return self.leaner5(x)
model=my_modle()

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    total=0
    correct=0
    running_loss = 0
    for i, data in enumerate(train_loader, 0):
        input, label = data
        # forward+backward+update
        output = model(input)
        label = label.type(torch.LongTensor)
        loss = criterion(output, label)

        _, predicted = torch.max(output.data, dim=1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 300 == 299:  # 每300轮输出一次
            print("[%d,%5d] loss:%.3f" % (epoch + 1, i + 1, running_loss / 300))
            running_loss = 0.0
    print("训练集上的准确率：%d%%"%(100*correct/total))


def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            images,labels=data
            output=model(images)
            _,predicted=torch.max(output.data,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print("测试集上的准确率:%d%%"%(100*correct/total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch % 10 == 9:
            test()


