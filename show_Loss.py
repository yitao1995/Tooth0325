import matplotlib.pyplot as plt
import numpy as np

point_set_whole = np.loadtxt('show_Loss.txt')  # .astype(np.float32)#序列
plt.plot(point_set_whole[800::8, 0], point_set_whole[800::8, 4])  # 绘制多组数据
#plt.plot(point_set_whole[300::8, 0], point_set_whole[300::8, 5])  # 绘制多组数据
#plt.plot(point_set_whole[300::8, 0], point_set_whole[300::8, 6])  # 绘制多组数据
plt.xlabel('epoch')
plt.ylabel('CD_loss')
plt.show()
