import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

#使中文字体显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False 

# 生成测试影像
print("生成测试影像...")
size = [64, 64, 32] #宽度64像素，高度64像素，深度32层
img = sitk.Image(size, sitk.sitkFloat32) #每个像素值32位浮点数
arr = np.zeros((32, 64, 64), dtype=np.float32)

# 创建椭球体
for z in range(32): #遍历深度
    for y in range(64): #遍历高度
        for x in range(64): #遍历宽度
            v = ((x-32)/20)**2 + ((y-32)/15)**2 + ((z-16)/10)**2
            arr[z,y,x] = 100*(1-v) + 10 if v <= 1 else 5

arr += np.random.normal(0, 3, arr.shape)  # 加噪声
arr = np.clip(arr, 0, 150)

img = sitk.GetImageFromArray(arr)
img.SetSpacing([1.0, 1.0, 2.0])

# 原始影像参数
print("\n")
print(f"原始影像:")
print(f"  尺寸: {img.GetSize()}, 间距: {img.GetSpacing()}")
print(f"  范围: [{arr.min():.2f}, {arr.max():.2f}], 均值: {arr.mean():.2f}, 标准差: {arr.std():.2f}")

# 归一化函数，三种归一化方法
def normalize(img, method):
    a = sitk.GetArrayFromImage(img)
    if method == 'zscore': #z- score标准化
        na = (a - a.mean()) / (a.std() + 1e-8)
        p = {'mean': a.mean(), 'std': a.std()}
    elif method == 'minmax': #Min-Max归一化
        na = (a - a.min()) / (a.max() - a.min() + 1e-8)
        p = {'min': a.min(), 'max': a.max()}
    elif method == 'percentile': #百分位数归一化
        p1, p99 = np.percentile(a, [1, 99])
        na = np.clip((a - p1) / (p99 - p1 + 1e-8), 0, 1)
        p = {'p1': p1, 'p99': p99}
    ni = sitk.GetImageFromArray(na.astype(np.float32))
    ni.CopyInformation(img)
    return ni, p

# 执行归一化并输出结果
print("\n" + "="*50)
for method in ['zscore', 'minmax', 'percentile']:
    ni, params = normalize(img, method)
    na = sitk.GetArrayFromImage(ni)
    
    print(f"\n{method.upper()} 归一化:")
    print(f"  参数: {params}")
    print(f"  范围: [{na.min():.4f}, {na.max():.4f}]")
    print(f"  均值: {na.mean():.4f}, 标准差: {na.std():.4f}")
    
    sitk.WriteImage(ni, f"normalized_{method}.nii")
    print(f"  保存: normalized_{method}.nii")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
mid = arr.shape[0]//2

axes[0,0].imshow(arr[mid], cmap='gray')
axes[0,0].set_title('原始'); axes[0,0].axis('off')

for i, m in enumerate(['zscore', 'minmax', 'percentile'], 1):
    na = sitk.GetArrayFromImage(sitk.ReadImage(f"normalized_{m}.nii"))
    axes[0,i].imshow(na[mid], cmap='gray')
    axes[0,i].set_title(m.upper()); axes[0,i].axis('off')

axes[1,0].hist(arr.flatten(), bins=30, alpha=0.5)
for i, m in enumerate(['zscore', 'minmax', 'percentile'], 1):
    na = sitk.GetArrayFromImage(sitk.ReadImage(f"normalized_{m}.nii"))
    axes[1,i].hist(na.flatten(), bins=30)

plt.tight_layout()
plt.savefig('normalization_result.png', dpi=120)
print(f"\n对比图: normalization_result.png")
plt.show()

print("\n完成!")