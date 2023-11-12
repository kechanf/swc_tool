import numpy as np

# 假设 images 是包含三维图像的列表
# 每个图像是一个三维的 NumPy 数组，例如 (depth, height, width)
images = [np.random.rand(5, 5, 5), np.random.rand(5, 5, 5), np.random.rand(5, 5, 5)]

# 使用 np.stack 将它们合并成一个四维的 NumPy 数组
# axis=0 表示在第0维度（样本）上进行合并
merged_images = np.stack(images, axis=0)

print(merged_images.shape)  # 输出合并后的数组形状