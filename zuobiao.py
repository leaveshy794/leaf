import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from numpy.ma.core import resize

# 加载一张图片
image = Image.open('test1.jpg')

image_np = np.array(image)

# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 在坐标轴上显示图片
ax.imshow(image_np)

# 定义矩形框的坐标(x1, y1, x2, y2)
# 这里的坐标是图片中矩形的左上角和右下角的坐标181, 46, 207, 54]], 'kaikou_Issue': [[0.839, 134, 58, 149, 66
x1, y1, x2, y2 =  134, 58, 149, 66 # 举例的坐标，您需要替换成实际的坐标

# 创建一个矩形框并添加到坐标轴上
rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

# 去掉坐标轴
ax.axis('off')

# 显示图形
plt.show()