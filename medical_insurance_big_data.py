import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 全局配置
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'],
    'axes.unicode_minus': False
})

def print_section(title, char='='):
    """打印章节标题"""
    print(f"\n{char * 60}\n{title}\n{char * 60}")

# 数据生成 
def generate_data(n=3000):
    """生成医保模拟数据"""
    np.random.seed(42)
    
    df = pd.DataFrame({
        '患者ID': [f'P{i:05d}' for i in range(1, n+1)],
        '性别': np.random.choice(['男', '女'], n, p=[0.48, 0.52]),
        '年龄': np.random.randint(18, 90, n),
        '地区': np.random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉'], n),
        '参保类型': np.random.choice(['职工医保', '居民医保', '新农合'], n, p=[0.5, 0.3, 0.2]),
        '就诊类型': np.random.choice(['门诊', '住院', '急诊'], n, p=[0.55, 0.3, 0.15]),
        '疾病类型': np.random.choice(['高血压', '糖尿病', '冠心病', '肺炎', '骨折', '胃病', '肿瘤'], n),
        '就诊月份': np.random.randint(1, 13, n)
    })
    
    # 根据就诊类型生成费用
    cost_params = {
        '门诊': (800, 500, 50),
        '急诊': (1500, 800, 100),
        '住院': (12000, 8000, 500)
    }
    
    costs = np.zeros(n)
    for idx, (_, row) in enumerate(df.iterrows()):
        mean, std, min_val = cost_params[row['就诊类型']]
        costs[idx] = max(min_val, np.random.normal(mean, std))
    
    df['医疗总费用'] = costs.round(2)
    df['报销比例'] = np.random.uniform(0.5, 0.9, n).round(2)
    df['报销金额'] = (df['医疗总费用'] * df['报销比例']).round(2)
    df['自付金额'] = (df['医疗总费用'] - df['报销金额']).round(2)
    
    return df

# 数据污染 
def inject_problems(df):
    """注入缺失值和异常值"""
    df_bad = df.copy()
    n = len(df_bad)
    
    # 缺失值注入
    for col in ['医疗总费用', '报销金额', '年龄']:
        idx = np.random.choice(n, int(n * 0.05), replace=False)
        df_bad.loc[idx, col] = np.nan
    
    # 异常值注入（费用列）
    for col in ['医疗总费用', '报销金额']:
        valid_idx = df_bad[col].notna()
        outlier_idx = np.random.choice(
            df_bad[valid_idx].index, 
            int(valid_idx.sum() * 0.03), 
            replace=False
        )
        df_bad.loc[outlier_idx, col] = np.random.choice(
            [-1000, np.random.uniform(500000, 2000000)], 
            len(outlier_idx)
        )
    
    # 年龄异常值
    idx = np.random.choice(n, 10, replace=False)
    df_bad.loc[idx, '年龄'] = np.random.choice([-5, 150, 200], 10)
    
    return df_bad

# 数据清洗 
def clean_data(df):
    """清洗数据：填充缺失值 + 剔除异常值"""
    df_clean = df.copy()
    
    # 1. 填充缺失值
    df_clean['年龄'] = df_clean['年龄'].fillna(df_clean['年龄'].median())
    df_clean['报销金额'] = df_clean['报销金额'].fillna(
        df_clean['医疗总费用'] * df_clean['报销比例']
    )
    df_clean['医疗总费用'] = df_clean['医疗总费用'].fillna(
        df_clean.groupby('就诊类型')['医疗总费用'].transform('median')
    )
    
    # 2. 剔除异常值（IQR方法）
    for col in ['医疗总费用', '报销金额', '自付金额']:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
        df_clean.loc[(df_clean[col] < lower) | (df_clean[col] > upper), col] = np.nan
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 3. 年龄异常修正
    df_clean.loc[(df_clean['年龄'] < 0) | (df_clean['年龄'] > 120), '年龄'] = df_clean['年龄'].median()
    
    return df_clean
    # 可视化
def plot_trends(df):
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 月度费用趋势（折线图）
    ax1 = plt.subplot(2, 3, 1)
    monthly = df.groupby('就诊月份')['医疗总费用'].mean()
    ax1.plot(monthly.index, monthly.values, marker='o', linewidth=2, color='steelblue', markersize=8)
    ax1.set_title('各月份平均医疗费用趋势', fontweight='bold')
    ax1.set_xlabel('月份'); ax1.set_ylabel('平均费用 (元)')
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(monthly.values):
        ax1.text(i+1, v+200, f'{v:.0f}', ha='center', fontsize=8)
    
    # 2. 年龄段费用趋势（折线图）
    ax2 = plt.subplot(2, 3, 2)
    bins = [18, 30, 40, 50, 60, 70, 90]
    labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '71+']
    age_group = pd.cut(df['年龄'], bins=bins, labels=labels)
    age_cost = df.groupby(age_group)['医疗总费用'].mean()
    ax2.plot(range(len(age_cost)), age_cost.values, marker='s', linewidth=2, color='coral', markersize=8)
    ax2.set_xticks(range(len(age_cost))); ax2.set_xticklabels(age_cost.index)
    ax2.set_title('不同年龄段平均医疗费用', fontweight='bold')
    ax2.set_xlabel('年龄段'); ax2.set_ylabel('平均费用 (元)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 疾病类型费用TOP8（水平柱状图）
    ax3 = plt.subplot(2, 3, 3)
    disease = df.groupby('疾病类型')['医疗总费用'].mean().sort_values(ascending=False).head(8)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(disease)))[::-1]
    bars = ax3.barh(disease.index, disease.values, color=colors, edgecolor='black')
    ax3.set_title('平均费用TOP8疾病类型', fontweight='bold')
    ax3.set_xlabel('平均费用 (元)')
    ax3.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(disease.values):
        ax3.text(v + 200, i, f'{v:.0f}', va='center', fontsize=9)
    
    # 4. 地区费用对比（垂直柱状图）
    ax4 = plt.subplot(2, 3, 4)
    region = df.groupby('地区')['医疗总费用'].mean().sort_values(ascending=False)
    colors4 = plt.cm.Reds(np.linspace(0.4, 0.9, len(region)))
    bars = ax4.bar(region.index, region.values, color=colors4, edgecolor='black')
    ax4.set_title('不同地区平均医疗费用', fontweight='bold')
    ax4.set_xlabel('地区'); ax4.set_ylabel('平均费用 (元)')
    ax4.tick_params(axis='x', rotation=15)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, region.values):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 100, f'{v:.0f}', ha='center', fontsize=8)
    
    # 5. 参保类型费用构成（分组柱状图）
    ax5 = plt.subplot(2, 3, 5)
    ins = df.groupby('参保类型')[['医疗总费用', '报销金额', '自付金额']].mean()
    x = np.arange(len(ins.index))
    width = 0.25
    ax5.bar(x - width, ins['医疗总费用'], width, label='总费用', color='steelblue')
    ax5.bar(x, ins['报销金额'], width, label='报销金额', color='coral')
    ax5.bar(x + width, ins['自付金额'], width, label='自付金额', color='lightgreen')
    ax5.set_xticks(x); ax5.set_xticklabels(ins.index)
    ax5.set_title('不同参保类型费用构成', fontweight='bold')
    ax5.set_ylabel('平均费用 (元)')
    ax5.legend(); ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 年龄×就诊类型热力图
    ax6 = plt.subplot(2, 3, 6)
    age_bins = pd.cut(df['年龄'], bins=[18, 30, 45, 60, 90], labels=['青年', '中年', '中老年', '老年'])
    pivot = df.pivot_table(values='医疗总费用', index=age_bins, columns='就诊类型', aggfunc='mean')
    im = ax6.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(range(len(pivot.columns))); ax6.set_xticklabels(pivot.columns)
    ax6.set_yticks(range(len(pivot.index))); ax6.set_yticklabels(pivot.index)
    ax6.set_title('年龄×就诊类型 费用热力图', fontweight='bold')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > pivot.values.max() / 2 else 'black'
            ax6.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=9)
    plt.colorbar(im, ax=ax6, label='平均费用 (元)')
    
    plt.tight_layout()
    plt.savefig('医保数据分析图.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存为 '医保数据分析图.png'")
    plt.show()

# 主流程 
def main():
    print_section("医保模拟数据分析")
    
    # 生成数据
    df_original = generate_data(3000)
    print(f"生成 {len(df_original)} 条记录")
    
    # 注入问题
    df_with_issues = inject_problems(df_original)
    print(f"\n缺失值总数: {df_with_issues.isnull().sum().sum()}")
    missing_cols = df_with_issues.isnull().sum()
    print(f"各列缺失情况:\n{missing_cols[missing_cols > 0]}")
    
    # 清洗数据
    print("\n数据清洗...")
    df_clean = clean_data(df_with_issues)
    print(f"清洗后缺失值: {df_clean.isnull().sum().sum()}")
    
    # 统计分析
    print_section("统计分析")
    print("\n描述性统计:")
    print(df_clean[['年龄', '医疗总费用', '报销金额', '自付金额']].describe().round(2))
    
    print("\n按就诊类型统计:")
    print(df_clean.groupby('就诊类型')[['医疗总费用', '报销金额']].mean().round(2))
    
    print("\n按参保类型统计:")
    print(df_clean.groupby('参保类型')[['医疗总费用', '报销金额']].mean().round(2))
    
    # 可视化
    print("\n生成可视化图表...")
    plot_trends(df_clean)
    
    # 质量报告
    print_section("数据质量报告")
    print(f"原始数据: {len(df_original)} 条")
    print(f"注入问题后: {len(df_with_issues)} 条 (缺失值 {df_with_issues.isnull().sum().sum()} 个)")
    print(f"清洗后: {len(df_clean)} 条 (缺失值 {df_clean.isnull().sum().sum()} 个)")
    print(f"\n总费用统计:")
    print(f"  均值: {df_clean['医疗总费用'].mean():.2f} 元")
    print(f"  中位数: {df_clean['医疗总费用'].median():.2f} 元")
    print(f"  最大值: {df_clean['医疗总费用'].max():.2f} 元")
    print(f"  最小值: {df_clean['医疗总费用'].min():.2f} 元")
    print("=" * 60)
    print("分析完成！")

if __name__ == "__main__":
    main()