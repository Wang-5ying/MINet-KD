import PIL.Image as Image
import cv2
import numpy as np
import os

root_path1 = '/home/wby/PycharmProjects/CoCA/data/images/'
root_path2 = '/home/wby/PycharmProjects/CoCA/data/depths/'
root_path3 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/w/r+d/0'
root_path4 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/w/r+d/1'
root_path5 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/w/r+d/2'
root_path6 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/w/r+d/3'
root_path7 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/w0/r+d/0'
root_path8 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/w0/r+d/1'
root_path9 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/w0/r+d/2'
root_path10 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/w0/r+d/3'
root_path11 = '/media/wby/shuju/four/four_result/Teacher'
root_path12 = '/media/wby/shuju/four/ablation_teacher_backbone+dsd'
save_path = '/home/wby/Desktop/0815'
root_1 = os.listdir(root_path11)
root_2 = []
for dic in root_1:
    root_2.append(os.listdir(os.path.join(root_path11, dic)))

for i in range(len(root_1)):
    path_1 = os.path.join(save_path, root_1[i])
    if not os.path.exists(path_1):
        os.makedirs(path_1)
    for j in range(len(root_2[i])):
        path_2 = os.path.join(path_1, root_2[i][j])
        if not os.path.exists(path_2):
            os.makedirs(path_2)

root_path = [root_path1, root_path3, root_path4, root_path5, root_path6, root_path11, root_path2, root_path7, root_path8, root_path9, root_path10, root_path12]
x = 6
y = 2
w, h = 320, 320

for i, r1 in enumerate(root_1):
    for j, r2 in enumerate(root_2[i]):
        dir_list = os.listdir(os.path.join(root_path11, r1, r2))
        for k in dir_list:
            k = k[:-4]
            img_list = []
            img_new = Image.new('RGB', (h * x, w * y), (255, 255, 255))
            for path in root_path:
                if 'images' in path and '183' not in r1:
                    k = k + '.jpg'
                    img = Image.open(os.path.join(path, r1, r2, k)).resize((w, h))
                    k = k[:-4]
                else:
                    if 'r+d' in path:
                        k = k + '.png.png'
                        img = Image.open(os.path.join(path, r1, r2, k)).resize((w, h))
                        k = k[:-8]
                    else:
                        k = k + '.png'
                        img = Image.open(os.path.join(path, r1, r2, k)).resize((w, h))
                        k = k[:-4]
                # img = Image.open(os.path.join(path, r1, r2, k)).resize((w, h))
                img_list.append(img)
            for m in range(y):
                for n in range(x):
                    img_new.paste(img_list[m * x + n], (n * h, m * w, (n + 1) * h, (m + 1) * w))
            img_new.save(os.path.join(save_path, r1, r2, f'{k}.png'))

