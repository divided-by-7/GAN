from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

# -----------------------------------------------------------------------
# 设置参数
train_set_rate = 0.8  # 设置训练集占训练集和验证集的比例
INPUT_SIZE = 10  # 生成器输入的噪声维度
LR = 0.0001  # 学习率
optimizer = 'RMSprop'  # optimizer
epoch_num = 200  # 迭代次数
batch_size = 32  # batch size
random_sample_num = 2000  # 在画图中G模型生成的样本点个数
CLAMP = 0.1
save_epoch = 5 # 每训练save_epoch次后绘制一次图并保存
m = loadmat("./points.mat")  # 数据集位置
# 默认保存路径：'./result/wgan'
# -----------------------------------------------------------------------

data_xx = m['xx']
np.random.shuffle(data_xx)

size = int(train_set_rate * len(data_xx))
train_set = data_xx[:size]
valid_set = data_xx[size:]

if not os.path.exists('./result/wgan'):
    os.mkdir('./result/wgan')


# 生成器
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.generator_model = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.generator_model(x)
        return x


# 判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        # 两层线性全连接
        self.discriminator_model = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.discriminator_model(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = discriminator().to(device)
G = generator().to(device)
d_optimizer = torch.optim.RMSprop(D.parameters(), lr=LR)
g_optimizer = torch.optim.RMSprop(G.parameters(), lr=LR)


def draw_scatter(data, color, x_min, x_max, y_min, y_max):
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Scatter Plot')
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=10)


# 画背景
def draw_background(D, x_min, x_max, y_min, y_max):
    i = x_min
    bg = []
    while i <= x_max - 0.01:
        j = y_min
        while j <= y_max - 0.01:
            bg.append([i, j])
            j += 0.01
        bg.append([i, y_max])
        i += 0.01
    j = y_min
    while j <= y_max - 0.01:
        bg.append([i, j])
        j += 0.01
        bg.append([i, y_max])
    bg.append([x_max, y_max])
    color = D(torch.Tensor(bg).to(device))
    bg = np.array(bg)
    cm = plt.cm.get_cmap('gray')
    sc = plt.scatter(bg[:, 0], bg[:, 1], c=np.squeeze(color.cpu().data), cmap=cm)
    # 显示颜色等级
    cb = plt.colorbar(sc)
    return cb


def test_G():
    G_input = torch.randn(random_sample_num, INPUT_SIZE).to(device)
    G_out = G(G_input)
    G_data = np.array(G_out.cpu().data)

    x_min = -1
    x_max = 2
    y_min = 0
    y_max = 1
    # 画背景
    cb = draw_background(D, x_min, x_max, y_min, y_max)
    # 画出测试集的点分布和生成器输出的点分布
    draw_scatter(valid_set, 'b', x_min, x_max, y_min, y_max)
    draw_scatter(G_data, 'r', x_min, x_max, y_min, y_max)
    return cb


for epoch in range(epoch_num):
    for i in range(int(size / batch_size)):
        label = torch.from_numpy(train_set[i * batch_size: (i + 1) * batch_size]).float().to(
            device)  # 把这个batch的训练集扔进gpu
        G_input = torch.randn(batch_size, INPUT_SIZE).to(device)  # 噪音扔进GPU
        G_out = G(G_input)  # 噪音预测的样本
        # 计算判别器判别的概率
        prob_gen = D(G_out)  # 使用判别器判别噪声生成的样本数据

        G_loss = - torch.mean(prob_gen).to(device)

        g_optimizer.zero_grad()
        G_loss.backward()
        g_optimizer.step()

        prob_label = D(label)  # 使用判别器判别真实数据的真假
        prob_gen = D(G_out.detach())  # 使用判别器判别噪声生成的样本数据

        D_loss = torch.mean(prob_gen - prob_label).to(device)

        d_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        d_optimizer.step()

        for p in D.parameters():
            p.data.clamp_(-CLAMP, CLAMP)

        print("epoch: %d \t batch: %d \t\t d_loss: %.8f \t g_loss: %.8f " % (epoch + 1, i + 1, D_loss, G_loss))

    if (epoch + 1) % save_epoch == 0:
        cb = test_G()
        plt.savefig('./result/wgan/epoch' + str(epoch + 1))
        cb.remove()
        plt.cla()
