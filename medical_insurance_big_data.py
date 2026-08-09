import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 使中文字体显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False 

print("=" * 60)
print("医保模拟数据分析")
print("=" * 60)

# 生成数据
def generate_data(n=3000):
    np.random.seed(42)
    
    # 生成基础数据
    df = pd.DataFrame({
        '患者ID': [f'P{i:05d}' for i in range(1, n+1)],
        '性别': np.random.choice(['男', '女'], n, p=[0.48, 0.52]),
        '年龄': np.random.randint(18, 90, n),
        '地区': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉'], n),
        '参保类型': np.random.choice(['职工医保', '居民医保', '新农合'], n, p=[0.5, 0.3, 0.2]),
        '就诊类型': np.random.choice(['门诊', '住院', '急诊'], n, p=[0.55, 0.3, 0.15]),
        '疾病类型': np.random.choice(['高血压', '糖尿病', '冠心病', '肺炎', '骨折', '胃病', '肿瘤'], n),
    })
    
    # 生成费用（根据就诊类型）
    costs = np.zeros(n)
    for i in range(n):
        if df.loc[i, '就诊类型'] == '门诊':
            costs[i] = max(50, np.random.normal(800, 500))
        elif df.loc[i, '就诊类型'] == '急诊':
            costs[i] = max(100, np.random.normal(1500, 800))
        else:
            costs[i] = max(500, np.random.normal(12000, 8000))
    
    df['医疗总费用'] = costs.round(2)
    df['报销比例'] = np.random.uniform(0.5, 0.9, n).round(2)
    df['报销金额'] = (df['医疗总费用'] * df['报销比例']).round(2)
    df['自付金额'] = (df['医疗总费用'] - df['报销金额']).round(2)
    df['就诊月份'] = np.random.randint(1, 13, n)
    
    return df

df_original = generate_data(3000)
print(f"生成 {len(df_original)} 条记录")

# 注入缺失值和异常值
print("\n注入缺失值和异常值...")

def inject_problems(df):
    df_bad = df.copy()
    n = len(df_bad)
    
    # 缺失值 
    for col in ['医疗总费用', '报销金额', '年龄']:
        idx = np.random.choice(n, int(n*0.05), replace=False)
        df_bad.loc[idx, col] = np.nan
    
    # 异常值 
    for col in ['医疗总费用', '报销金额']:
        valid = df_bad[col].notna()
        idx = df_bad[valid].index
        outlier_idx = np.random.choice(idx, int(len(idx)*0.03), replace=False)
        for i in outlier_idx:
            df_bad.loc[i, col] = np.random.choice([-1000, np.random.uniform(500000, 2000000)])
    
    # 年龄异常
    idx = np.random.choice(n, 10, replace=False)
    df_bad.loc[idx, '年龄'] = np.random.choice([-5, 150, 200])
    
    return df_bad

df_with_issues = inject_problems(df_original)
print(f"缺失值总数: {df_with_issues.isnull().sum().sum()}")
print(f"各列缺失情况:\n{df_with_issues.isnull().sum()[df_with_issues.isnull().sum() > 0]}")

# 数据清洗
print("\n数据清洗...")

def clean_data(df):
    df_clean = df.copy()
    
    # 填充缺失值
    df_clean['年龄'] = df_clean['年龄'].fillna(df_clean['年龄'].median())
    df_clean['报销金额'] = df_clean['报销金额'].fillna(df_clean['医疗总费用'] * df_clean['报销比例'])
    df_clean['医疗总费用'] = df_clean['医疗总费用'].fillna(df_clean.groupby('就诊类型')['医疗总费用'].transform('median'))
    
    # 剔除异常值（IQR方法）
    for col in ['医疗总费用', '报销金额', '自付金额']:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
        df_clean.loc[(df_clean[col] < lower) | (df_clean[col] > upper), col] = np.nan
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 年龄异常修正
    df_clean.loc[(df_clean['年龄'] < 0) | (df_clean['年龄'] > 120), '年龄'] = df_clean['年龄'].median()
    
    return df_clean

df_clean = clean_data(df_with_issues)
print(f"清洗后缺失值: {df_clean.isnull().sum().sum()}")

# 统计分析 
print("\n统计分析...")

print("\n描述性统计:")
print(df_clean[['年龄', '医疗总费用', '报销金额', '自付金额']].describe().round(2))

print("\n按就诊类型统计:")
print(df_clean.groupby('就诊类型')[['医疗总费用', '报销金额']].mean().round(2))

print("\n按参保类型统计:")
print(df_clean.groupby('参保类型')[['医疗总费用', '报销金额']].mean().round(2))

# 可视化 
print("\n生成可视化图表...")

fig = plt.figure(figsize=(15, 10))

# 折线图 - 各月份医疗总费用趋势
ax1 = plt.subplot(2, 3, 1)
monthly_cost = df_clean.groupby('就诊月份')['医疗总费用'].mean()
ax1.plot(monthly_cost.index, monthly_cost.values, marker='o', linewidth=2, color='steelblue', markersize=8)
ax1.set_title('各月份平均医疗费用趋势', fontsize=12, fontweight='bold')
ax1.set_xlabel('月份')
ax1.set_ylabel('平均费用 (元)')
ax1.grid(True, alpha=0.3)
for i, v in enumerate(monthly_cost.values):
    ax1.text(i+1, v+200, f'{v:.0f}', ha='center', fontsize=8)

# 折线图 - 不同年龄段的费用趋势
ax2 = plt.subplot(2, 3, 2)
age_bins = pd.cut(df_clean['年龄'], bins=[18, 30, 40, 50, 60, 70, 90], labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71+'])
age_cost = df_clean.groupby(age_bins)['医疗总费用'].mean()
ax2.plot(range(len(age_cost)), age_cost.values, marker='s', linewidth=2, color='coral', markersize=8)
ax2.set_xticks(range(len(age_cost)))
ax2.set_xticklabels(age_cost.index)
ax2.set_title('不同年龄段平均医疗费用', fontsize=12, fontweight='bold')
ax2.set_xlabel('年龄段')
ax2.set_ylabel('平均费用 (元)')
ax2.grid(True, alpha=0.3)

# 柱状图 - 各疾病类型平均费用TOP8
ax3 = plt.subplot(2, 3, 3)
disease_cost = df_clean.groupby('疾病类型')['医疗总费用'].mean().sort_values(ascending=False).head(8)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(disease_cost)))[::-1]
bars = ax3.barh(disease_cost.index, disease_cost.values, color=colors, edgecolor='black')
ax3.set_title('平均费用TOP8疾病类型', fontsize=12, fontweight='bold')
ax3.set_xlabel('平均费用 (元)')
ax3.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(disease_cost.values):
    ax3.text(v + 200, i, f'{v:.0f}', va='center', fontsize=9)

# 柱状图 - 不同地区费用对比
ax4 = plt.subplot(2, 3, 4)
region_cost = df_clean.groupby('地区')['医疗总费用'].mean().sort_values(ascending=False)
colors4 = plt.cm.Reds(np.linspace(0.4, 0.9, len(region_cost)))
bars = ax4.bar(region_cost.index, region_cost.values, color=colors4, edgecolor='black')
ax4.set_title('不同地区平均医疗费用', fontsize=12, fontweight='bold')
ax4.set_xlabel('地区')
ax4.set_ylabel('平均费用 (元)')
ax4.tick_params(axis='x', rotation=15)
ax4.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, region_cost.values):
    ax4.text(bar.get_x() + bar.get_width()/2, v + 100, f'{v:.0f}', ha='center', fontsize=8)

# 柱状图 - 不同参保类型费用构成
ax5 = plt.subplot(2, 3, 5)
insurance_group = df_clean.groupby('参保类型')[['医疗总费用', '报销金额', '自付金额']].mean()
x = np.arange(len(insurance_group.index))
width = 0.25
ax5.bar(x - width, insurance_group['医疗总费用'], width, label='总费用', color='steelblue')
ax5.bar(x, insurance_group['报销金额'], width, label='报销金额', color='coral')
ax5.bar(x + width, insurance_group['自付金额'], width, label='自付金额', color='lightgreen')
ax5.set_xticks(x)
ax5.set_xticklabels(insurance_group.index)
ax5.set_title('不同参保类型费用构成', fontsize=12, fontweight='bold')
ax5.set_ylabel('平均费用 (元)')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 热力图 - 年龄与就诊类型的费用矩阵
ax6 = plt.subplot(2, 3, 6)
# 创建年龄分组
age_groups = pd.cut(df_clean['年龄'], bins=[18, 30, 45, 60, 90], labels=['青年', '中年', '中老年', '老年'])
pivot_table = df_clean.pivot_table(values='医疗总费用', index=age_groups, columns='就诊类型', aggfunc='mean')
# 绘制热力图
im = ax6.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
ax6.set_xticks(range(len(pivot_table.columns)))
ax6.set_xticklabels(pivot_table.columns)
ax6.set_yticks(range(len(pivot_table.index)))
ax6.set_yticklabels(pivot_table.index)
ax6.set_title('年龄×就诊类型 费用热力图', fontsize=12, fontweight='bold')
# 添加数值标注
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        ax6.text(j, i, f'{pivot_table.values[i, j]:.0f}', ha='center', va='center', 
                color='white' if pivot_table.values[i, j] > pivot_table.values.max()/2 else 'black', fontsize=9)
plt.colorbar(im, ax=ax6, label='平均费用 (元)')

plt.tight_layout()
plt.savefig('医保数据分析图.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存为 '医保数据分析图.png'")
plt.show()

# 总结报告 
print("\n【6】数据质量报告")
print("=" * 60)
print(f"原始数据: {len(df_original)} 条")
print(f"注入问题后: {len(df_with_issues)} 条 (缺失值 {df_with_issues.isnull().sum().sum()} 个)")
print(f"清洗后: {len(df_clean)} 条 (缺失值 {df_clean.isnull().sum().sum()} 个)")
print(f"\n总费用统计:")
print(f" 均值: {df_clean['医疗总费用'].mean():.2f} 元")
print(f" 中位数: {df_clean['医疗总费用'].median():.2f} 元")
print(f" 最大值: {df_clean['医疗总费用'].max():.2f} 元")
print(f" 最小值: {df_clean['医疗总费用'].min():.2f} 元")
print("=" * 60)
print("分析完成！")