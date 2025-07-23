import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入自定义模块
import tool.dataload_seg as d
from model.cdnet_newdic import ChangeDetector, Load_Weight_FordataParallel

# if Dataset == 'WHU':
#     DATA_PATH = r'C:\Users\Administrator\Desktop\CSINet\WHU-CD'
#     MASK_PATH = r'C:\segment-anything-main\segment_anything\pred\WHU-CD'
# if Dataset == 'GZ':
#     DATA_PATH = r'C:\Users\Administrator\Desktop\CSINet\GZ-CD'
#     MASK_PATH = r'C:\segment-anything-main\segment_anything\pred\GZ-CD'
# if Dataset == 'LEVIR':
#     DATA_PATH = r'C:\Users\Administrator\Desktop\CSINet\LEVIR-CD'
#     MASK_PATH = r'C:\segment-anything-main\segment_anything\pred\LEVIR-CD'


# 设置数据集路径
Dataset = "WHU"
DATA_PATH = r'/home/xietong/project/ws/FDLdet/NJWH2012_2019(1)/NJWH2012_2019'
MASK_PATH = r'./pred/NJWH-CD'
TEST_DATA_PATH = os.path.join(DATA_PATH)
TEST_LABEL_PATH = os.path.join(DATA_PATH)
TEST_TXT_PATH = os.path.join(TEST_DATA_PATH, 'list', 'test.txt')
MASK_PATH = os.path.join(MASK_PATH)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载测试数据集
seg_num = 50
test_batch_size = 1
test_data = d.Dataset(TEST_DATA_PATH, TEST_LABEL_PATH, TEST_TXT_PATH, MASK_PATH, 'njwh', transform=False,
                      seg_num=seg_num, ratio=2)
test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

# 加载模型
Channel_num = 64
word_num = 64
key = 'res1'
model = ChangeDetector(Channel_num, 2, key, word_num=word_num, backbone='resnet152')
model = torch.nn.DataParallel(model)
model = model.to(device)

# 加载训练好的模型权重
Log_path = 'results/' + 'GZ' + '/'
state_dict = torch.load(Log_path + 'Best_model_cdnewdic_lossall200_fastsam_res50.pth')
state_dict = Load_Weight_FordataParallel(state_dict, need_dataparallel=1)
model.load_state_dict(state_dict)
model.eval()

# 创建保存预测结果的目录
prediction_dir = 'NJWH2012_2019_GZmodel'
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)


# 测试模型并保存预测结果
def test_model(data_loader, model, device):
    model.eval()
    with torch.no_grad():
        for i, (imgt1, imgt2, gt, filename, _, seg1, num1, max_num1, seg2, num2, max_num2,mask1,mask2) in enumerate(
                tqdm(data_loader)):
            imgt1, imgt2, gt ,mask1,mask2= imgt1.cuda(), imgt2.cuda(), gt.cuda(),mask1.cuda(),mask2.cuda()
            seg1, num1, max_num1, seg2, num2, max_num2 = seg1.to(device), num1.to(device), max_num1.to(device), seg2.to(
                device), num2.to(device), max_num2.to(device)
            # print(filename[i].split('/')[-1].split('.')[0])
            batch_size = imgt1.shape[0]

            for j in range(batch_size):
                prefix = filename[j].split('/')[-1].split('.')[0]
            prediction = model(imgt1, imgt2, seg1, num1, max_num1, seg2, num2, max_num2,mask1,mask2,prefix=prefix)
            prediction = torch.max(prediction, dim=1)[1]

            # 保存预测结果
            batch_size = imgt1.shape[0]
            for j in range(batch_size):
                pr_name = filename[j].split('/')[-1].split('.')[0] + '.png'
                plt.imsave(os.path.join(prediction_dir, pr_name), prediction[j].detach().cpu(), cmap='gray')


if __name__ == "__main__":
    test_model(test_loader, model, device)
