#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^引入相关包^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch.optim as optim
import xlrd
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import random

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^划分数据集^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
normalize = transforms.Normalize(mean=[0.35641459,0.40738414,0.38418854], std=[0.19435959,0.16657008,0.18240114])#根据给出样本计算均值和方差
#数据处理
train_transformer_ImageNet = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.Resize((64,64),Image.LANCZOS),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize
])
val_transformer_ImageNet = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.Resize((64, 64), Image.LANCZOS),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize
])

#定义数据集
class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        shape=image.size
        if shape[0] < shape[1]:#统一照片方向
            image = image.transpose(Image.ROTATE_90)  # 旋转
            image = image.resize((112, 64), Image.LANCZOS)  # 统一尺寸
        image = self.transform(image)
        return image, self.labels[idx]

#根据正反例的比例划分训练集和验证集中正反例的个数
def split_Train_Test_Data(data_dir, ratio):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)  # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)

    train_inputs, test_inputs= [], []
    train_labels,test_labels = [], []
    for i, data in enumerate(character):
        num_sample_train = int(len(data) * ratio[0])
        random.shuffle(data)  # 打乱后抽取
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:]:
            test_inputs.append(str(x))
            test_labels.append(i)
    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer_ImageNet),
                                  batch_size=80, shuffle=True)
    test_dataloader = DataLoader(MyDataset(test_inputs, test_labels, val_transformer_ImageNet),
                                batch_size=80, shuffle=False)

    return train_dataloader, test_dataloader

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^读取《比赛评分计算辅助表》中的正确标签^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def extract(inpath):
    list1 = []
    data = xlrd.open_workbook(inpath, encoding_override='utf-8')
    table = data.sheets()[0]  # 选定表
    nrows = table.nrows  # 获取行号
    ncols = table.ncols  # 获取列号

    for i in range(1, nrows):  # 第0行为表头
        alldata = table.row_values(i)  # 循环输出excel表中每一行，即所有数据
        result = alldata[2]  # 取出表中第二列数据
        list1.append(int(result))
    return  list1
inpath = 'C:\\Users\\yxh\\Desktop\\比赛评分计算辅助表.xls'  # excel文件所在路径
list1=extract(inpath)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^引入数据集^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
data_dir = 'G:\\期末大作业\\sample'
train_dataloader, test_dataloader = split_Train_Test_Data(data_dir, [0.9,0.1])

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^启用GPU^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^加载模型(GooleNet)^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
net = models.googlenet(pretrained=True)
net.fc = nn.Linear(1024,2,bias=True)
net.to(device)  # net into cuda

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^定义超参数^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
EPOCH = 30#遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
LR = 0.001        #学习率

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^定义损失函数和优化方式^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=0.0005) #优化方式为Adam

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^训练^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
if __name__ == "__main__":
    acc = 0
    pre = 0
    rec = 0
    f = 0
    all = 0
    print("Start Training, 微调GooleNet!")
    with open("微调GoogleNet.txt", "w",encoding='utf-8')as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                sum_loss = 0.0
                for i, data in enumerate(train_dataloader):
                    inputs, labels = data
                    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                    optimizer.zero_grad()  # 将梯度归零
                    outputs = net(inputs)  # 将数据传入网络进行前向运算
                    loss = criterion(outputs, labels)  # 得到损失函数
                    loss.backward()  # 反向传播
                    optimizer.step()  # 通过梯度做一步参数更新
                    sum_loss += loss.item()
                    if i % 10 == 9:
                        print('[%d,%d] loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))
                        f2.write('[%d,%d] loss:%.03f' % (epoch + 1, i + 1, sum_loss / 100))
                        f2.write('\n')
                        f2.flush()
                        sum_loss = 0.0

            net.eval()  # 把模型转为test模式
            correct = 0
            total = 0
            test_list1=[]
            test_list2=[]
            test_listAD=[]
            test_listAD2=[]
            test_list3=[]
            for i,data_test in enumerate(test_dataloader):
                images, labels = data_test
                test_list1.append(labels.tolist())
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
                output_test = net(images)
                _, predicted = torch.max(output_test, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                test_listAD.append(predicted.tolist())
            for i in range(len(test_list1)):
                for j in range(len(test_list1[i])):
                    test_list2.append(test_list1[i][j])
            for i in range(len(test_listAD)):
                for j in range(len(test_listAD[i])):
                    test_listAD2.append(test_listAD[i][j])
            for n in range(len(test_listAD2)):
                if test_list2[n]==test_listAD2[n] and test_list2[n]==1:
                    test_list3.append(1)
                else:
                    test_list3.append(0)
            test_pre=sum(test_list3)/sum(test_listAD2)#计算精确率
            test_rec=sum(test_list3)/sum(test_list2)#计算召回率
            test_f=(2*test_pre*test_rec)/(test_pre+test_rec)#计算F值
            test_all=40*(correct.item() / total+test_f)#总分
            print("correct1: ", correct)
            print("Test acc: {0}%".format(100 * correct.item() / total))
            print("精确率", test_pre)
            print("召回率", test_rec)
            print("F值", test_f)
            print("总分",test_all)

            #开始预测
            print("对比开始!")
            file = open("test2.csv", "w", newline="")
            # 创建文件，分别是文件名、w打开方式(w代表新建，如果已存在，就删除重写)、newline(如果不加，每行数据就会多一空白行)

            fwrite = csv.writer(file)
            # 获取写文件的对象

            fwrite.writerow(["id", "labels"])
            # 写入标题头

            list_ad = []#测试值
            test = []#全部对比结果
            test2 = []#正样本对比结果
            for i in range(0, 1390):
                img = Image.open('G:\\test\\' + str(i) + ".png").convert('RGB')  # 读取要预测的图片
                shape = img.size
                if shape[0] < shape[1]:#统一照片方向
                    img = img.transpose(Image.ROTATE_90)  # 旋转
                    img = img.resize((112, 64), Image.LANCZOS)  # 改变尺寸
                trans = transforms.Compose(
                    [
                        transforms.Resize((64, 64), Image.LANCZOS),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.35641459, 0.40738414, 0.38418854],
                                             std=[0.19435959, 0.16657008, 0.18240114])
                    ])
                img = trans(img)
                img = img.to(device)
                img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
                output = net(img)
                prob = F.softmax(output, dim=1)
                prob = Variable(prob)
                prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
                pred = np.argmax(prob)  # 选出概率最大的一个
                list_ad.append(pred)
                fwrite.writerow([i, pred])
            for n in range(len(list_ad)):
                if list1[n]==list_ad[n]:
                    test.append(1)
                else:
                    test.append(0)
                if list1[n]==list_ad[n] and list_ad[n]==1:
                    test2.append(1)
                else:
                    test2.append(0)
            acc=sum(test)/1390#计算准确度
            pre=sum(test2)/sum(list_ad)#计算精确率
            rec=sum(test2)/210#计算召回率
            f=(2*pre*rec)/(pre+rec)#计算F值
            all=40*(acc+f)#总分
            print("准确度",acc)
            print("精确率", pre)
            print("召回率", rec)
            print("F值", f)
            print("总分",all)
            file.close()
    torch.save(net, "微调GoogleNet(%f).pth"%(all))
    print("Training Finished, TotalEPOCH=%d" % EPOCH)
