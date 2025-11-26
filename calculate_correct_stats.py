import pandas as pd
import numpy as np
from scipy import stats

# Load model comparison data
df = pd.read_csv('outputs/results/model_comparison.csv')

# Filter valid comparisons (exclude BTC-USD which has no TimeGAN data)
valid_df = df.dropna(subset=['TimeGAN_MeanDiff', 'Diffusion_MeanDiff'])

print("="*80)
print("CORRECT STATISTICS FOR REPORT")
print("="*80)

print(f"\nValid comparisons: {len(valid_df)}/12 assets")
print(f"Assets: {', '.join(valid_df['Asset'].tolist())}")

# TimeGAN statistics
tg_mean = valid_df['TimeGAN_MeanDiff'].mean()
tg_std = valid_df['TimeGAN_MeanDiff'].std()
tg_median = valid_df['TimeGAN_MeanDiff'].median()

print(f"\n{'='*80}")
print("TIMEGAN MEAN DIFFERENCE:")
print(f"{'='*80}")
print(f"Mean:   {tg_mean:.6f} (should be: {tg_mean:.3f} ± {tg_std:.3f})")
print(f"Std:    {tg_std:.6f}")
print(f"Median: {tg_median:.6f}")
print(f"Min:    {valid_df['TimeGAN_MeanDiff'].min():.6f} ({valid_df.loc[valid_df['TimeGAN_MeanDiff'].idxmin(), 'Asset']})")
print(f"Max:    {valid_df['TimeGAN_MeanDiff'].max():.6f} ({valid_df.loc[valid_df['TimeGAN_MeanDiff'].idxmax(), 'Asset']})")

# Diffusion statistics
diff_mean = valid_df['Diffusion_MeanDiff'].mean()
diff_std = valid_df['Diffusion_MeanDiff'].std()
diff_median = valid_df['Diffusion_MeanDiff'].median()

print(f"\n{'='*80}")
print("DIFFUSION MEAN DIFFERENCE:")
print(f"{'='*80}")
print(f"Mean:   {diff_mean:.6f} (should be: {diff_mean:.3f} ± {diff_std:.3f})")
print(f"Std:    {diff_std:.6f}")
print(f"Median: {diff_median:.6f}")
print(f"Min:    {valid_df['Diffusion_MeanDiff'].min():.6f} ({valid_df.loc[valid_df['Diffusion_MeanDiff'].idxmin(), 'Asset']})")
print(f"Max:    {valid_df['Diffusion_MeanDiff'].max():.6f} ({valid_df.loc[valid_df['Diffusion_MeanDiff'].idxmax(), 'Asset']})")

# Improvement calculation
improvement_abs = diff_mean - tg_mean
improvement_pct = (improvement_abs / diff_mean) * 100

print(f"\n{'='*80}")
print("IMPROVEMENT:")
print(f"{'='*80}")
print(f"Absolute difference: {improvement_abs:.6f}")
print(f"Percentage improvement: {improvement_pct:.2f}% (TimeGAN better)")
print(f"Relative improvement: {-(improvement_abs/tg_mean)*100:.1f}%")

# Winner counts
winner_counts = valid_df['MeanDiff_Winner'].value_counts()
print(f"\n{'='*80}")
print("WINNER BREAKDOWN:")
print(f"{'='*80}")
for winner, count in winner_counts.items():
    print(f"{winner}: {count}/{len(valid_df)} ({count/len(valid_df)*100:.1f}%)")

# Paired t-test
t_stat, p_value = stats.ttest_rel(valid_df['TimeGAN_MeanDiff'], valid_df['Diffusion_MeanDiff'])
print(f"\n{'='*80}")
print("PAIRED T-TEST:")
print(f"{'='*80}")
print(f"t-statistic: {t_stat:.6f}")
print(f"p-value: {p_value:.6f}")
if p_value < 0.001:
    print(f"Result: *** p < 0.001 (highly significant)")
elif p_value < 0.01:
    print(f"Result: ** p < 0.01 (very significant)")
elif p_value < 0.05:
    print(f"Result: * p < 0.05 (significant)")
else:
    print(f"Result: Not significant (p >= 0.05)")

# Cohen's d
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (x.mean() - y.mean()) / np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / dof)

cohen_d = cohens_d(valid_df['TimeGAN_MeanDiff'], valid_df['Diffusion_MeanDiff'])
print(f"\n{'='*80}")
print("COHEN'S D (EFFECT SIZE):")
print(f"{'='*80}")
print(f"Cohen's d: {cohen_d:.4f}")
if abs(cohen_d) < 0.2:
    effect = "small"
elif abs(cohen_d) < 0.5:
    effect = "medium"
else:
    effect = "large"
print(f"Effect size: {effect}")

# Diffusion KS statistics
ks_mean = valid_df['Diffusion_KS'].mean()
ks_std = valid_df['Diffusion_KS'].std()

print(f"\n{'='*80}")
print("DIFFUSION KS STATISTICS:")
print(f"{'='*80}")
print(f"Mean:   {ks_mean:.6f} (should be: {ks_mean:.3f} ± {ks_std:.3f})")
print(f"Std:    {ks_std:.6f}")

# Wilcoxon signed-rank test (non-parametric)
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(valid_df['TimeGAN_MeanDiff'], valid_df['Diffusion_MeanDiff'])
print(f"\n{'='*80}")
print("WILCOXON SIGNED-RANK TEST (non-parametric):")
print(f"{'='*80}")
print(f"Statistic: {wilcoxon_stat:.4f}")
print(f"p-value: {wilcoxon_p:.6f}")

print(f"\n{'='*80}")
print("LATEX TABLE CORRECTIONS:")
print(f"{'='*80}")
print(f"TimeGAN Mean Difference:     ${tg_mean:.3f} \\pm {tg_std:.3f}$")
print(f"Diffusion Mean Difference:   ${diff_mean:.3f} \\pm {diff_std:.3f}$")
print(f"Relative Improvement:        {improvement_pct:.1f}\\%")
print(f"KS Statistic:                ${ks_mean:.3f} \\pm {ks_std:.3f}$")
print(f"Winner Count (TimeGAN):      {winner_counts.get('TimeGAN', 0)}/{len(valid_df)} ({winner_counts.get('TimeGAN', 0)/len(valid_df)*100:.1f}\\%)")
print(f"Winner Count (Tie):          {winner_counts.get('Tie', 0)}/{len(valid_df)} ({winner_counts.get('Tie', 0)/len(valid_df)*100:.1f}\\%)")
print(f"p-value:                     {p_value:.4f}***")
print(f"Cohen's d:                   {cohen_d:.2f}")

print(f"\n{'='*80}")
print("ABSTRACT CORRECTIONS:")
print(f"{'='*80}")
print(f"Old: 'TimeGAN achieves significantly better performance (mean difference: 0.060 ± 0.019)'")
print(f"New: 'TimeGAN achieves significantly better performance (mean difference: {tg_mean:.3f} ± {tg_std:.3f})'")
print(f"")
print(f"Old: 'compared to Diffusion Models (0.130 ± 0.025)'")
print(f"New: 'compared to Diffusion Models ({diff_mean:.3f} ± {diff_std:.3f})'")
print(f"")
print(f"Old: 'p=0.0004, Cohen's d=-2.21'")
print(f"New: 'p={p_value:.4f}, Cohen's d={cohen_d:.2f}'")
print(f"")
print(f"Old: 'average: 0.388'")
print(f"New: 'average: {ks_mean:.3f}'")

print(f"\n{'='*80}")
