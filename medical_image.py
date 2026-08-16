import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 生成椭球体测试影像
size = (64, 64, 32)
arr = np.zeros((32, 64, 64), dtype=np.float32)
for z in range(32):
    for y in range(64):
        for x in range(64):
            v = ((x-32)/20)**2 + ((y-32)/15)**2 + ((z-16)/10)**2
            arr[z, y, x] = 100*(1-v) + 10 if v <= 1 else 5
arr += np.random.normal(0, 3, arr.shape)
arr = np.clip(arr, 0, 150)

img = sitk.GetImageFromArray(arr)
img.SetSpacing([1.0, 1.0, 2.0])

print(f"原始影像: 尺寸 {img.GetSize()}, 间距 {img.GetSpacing()}, "
      f"范围 [{arr.min():.2f}, {arr.max():.2f}], 均值 {arr.mean():.2f}, 标准差 {arr.std():.2f}\n")

# 归一化函数
def normalize(img, method):
    a = sitk.GetArrayFromImage(img)
    if method == 'zscore':
        na = (a - a.mean()) / (a.std() + 1e-8)
        params = {'mean': a.mean(), 'std': a.std()}
    elif method == 'minmax':
        na = (a - a.min()) / (a.max() - a.min() + 1e-8)
        params = {'min': a.min(), 'max': a.max()}
    elif method == 'percentile':
        p1, p99 = np.percentile(a, [1, 99])
        na = np.clip((a - p1) / (p99 - p1 + 1e-8), 0, 1)
        params = {'p1': p1, 'p99': p99}
    ni = sitk.GetImageFromArray(na.astype(np.float32))
    ni.CopyInformation(img)
    return ni, params

# 执行归一化
methods = ['zscore', 'minmax', 'percentile']
results = {}
for method in methods:
    ni, params = normalize(img, method)
    na = sitk.GetArrayFromImage(ni)
    results[method] = na
    sitk.WriteImage(ni, f"normalized_{method}.nii")
    print(f"{method.upper():12} 参数 {params} 范围 [{na.min():.4f}, {na.max():.4f}] "
          f"均值 {na.mean():.4f} 标准差 {na.std():.4f}")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
mid = arr.shape[0] // 2

# 第一行：图像
axes[0, 0].imshow(arr[mid], cmap='gray')
axes[0, 0].set_title('原始')
axes[0, 0].axis('off')
for i, method in enumerate(methods, 1):
    axes[0, i].imshow(results[method][mid], cmap='gray')
    axes[0, i].set_title(method.upper())
    axes[0, i].axis('off')

# 第二行：直方图
axes[1, 0].hist(arr.flatten(), bins=30, alpha=0.5)
for i, method in enumerate(methods, 1):
    axes[1, i].hist(results[method].flatten(), bins=30)

plt.tight_layout()
plt.savefig('normalization_result.png', dpi=120)
plt.show()
print("\n完成! 对比图已保存为 normalization_result.png")