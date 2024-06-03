import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

sns.set()

norm = np.random.normal(0, 1, 20) # 正常データ
anom = np.array([-3, 3]) # 異常データ
plt.scatter(norm, [0]*len(norm), color='green', label='normal')
plt.scatter(anom, [0]*len(anom), color='red', label='anomaly')

# 最尤推定
N = len(norm)
mu_hat = sum(norm)/N
s_hat = sum([(x-mu_hat)**2 for x in norm])/N

# カイ二乗分布の分位点を閾値とする
a_th_5 = chi2.ppf(q=0.95, df=1)
a_th_30 = chi2.ppf(q=0.7, df=1)

# 各データの異常度を計算
data = np.hstack([norm, anom])
anom_score = []
for d in data:
    score = ((d - mu_hat)/s_hat)**2
    anom_score.append(score)

#描画
norm_pred_5 = [d for d,a in zip(data, anom_score) if a<=a_th_5]
anom_pred_5 = [d for d,a in zip(data, anom_score) if a>a_th_5]
norm_pred_30 = [d for d,a in zip(data, anom_score) if a<=a_th_30]
anom_pred_30 = [d for d,a in zip(data, anom_score) if a>a_th_30]

fig, axes = plt.subplots(1,3, figsize=(15,5))

axes[0].scatter(norm, [0]*len(norm), color='green', label='normal')
axes[0].scatter(anom, [0]*len(anom), color='red', label='anomaly')
axes[0].set_title('Truth')
axes[1].scatter(norm_pred_5, [0]*len(norm_pred_5), color='green', label='normal pred')
axes[1].scatter(anom_pred_5, [0]*len(anom_pred_5), color='red', label='anomaly pred')
axes[1].set_title('Threshold:5%')
axes[2].scatter(norm_pred_30, [0]*len(norm_pred_30), color='green', label='normal pred')
axes[2].scatter(anom_pred_30, [0]*len(anom_pred_30), color='red', label='anomaly pred')
axes[2].set_title('Threshold:30%')
plt.legend()