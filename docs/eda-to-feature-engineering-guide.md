# EDA → Feature Engineering Guide
## Home Credit Default Risk

> Tài liệu này mô tả quy trình từng bước: từ EDA để hiểu data, đến các quyết định feature engineering có căn cứ. Mỗi bước EDA đều kết thúc bằng một câu hỏi cụ thể cần trả lời — câu trả lời đó dẫn trực tiếp đến quyết định engineering.

---

## Tổng quan quy trình

```
Bước 1: Hiểu cấu trúc tổng thể
    ↓
Bước 2: EDA bảng chính (application_train)
    ↓
Bước 3: EDA từng bảng phụ
    ↓
Bước 4: Quyết định aggregation strategy
    ↓
Bước 5: Tạo engineered features (ratio, interaction)
    ↓
Bước 6: Feature selection
    ↓
Bước 7: Final feature set → Model
```

---

## Bước 1 — Hiểu cấu trúc tổng thể

**Mục tiêu**: Trước khi nhìn vào bất kỳ feature nào, cần hiểu mối quan hệ giữa các bảng và kích thước của chúng.

### Code khởi đầu

```python
import pandas as pd
import numpy as np

files = {
    'application_train': 'application_train.csv',
    'bureau':            'bureau.csv',
    'bureau_balance':    'bureau_balance.csv',
    'previous_app':      'previous_application.csv',
    'pos_cash':          'POS_CASH_balance.csv',
    'installments':      'installments_payments.csv',
    'credit_card':       'credit_card_balance.csv',
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(f"{name:20s}: {df.shape[0]:>8,} rows × {df.shape[1]:>3} cols")
```

### Những gì cần ghi lại

- Số rows mỗi bảng → bảng nào "nặng" nhất (POS_CASH thường là lớn nhất ~10M rows)
- Ratio rows/unique SK_ID_CURR mỗi bảng phụ → biết trung bình mỗi khách hàng có bao nhiêu records

```python
app = pd.read_csv('application_train.csv')
n_customers = app['SK_ID_CURR'].nunique()  # ~307k

for name, path in files.items():
    if name == 'application_train':
        continue
    df = pd.read_csv(path)
    n_unique = df['SK_ID_CURR'].nunique()
    coverage = n_unique / n_customers * 100
    ratio = len(df) / n_unique if n_unique > 0 else 0
    print(f"{name:20s}: coverage={coverage:.1f}%  avg_rows_per_customer={ratio:.1f}")
```

**Câu hỏi cần trả lời**: Bảng nào có coverage thấp (< 70%)? → Khách hàng không có records ở bảng đó sẽ nhận NaN sau join — cần strategy riêng.

---

## Bước 2 — EDA bảng chính (application_train)

### 2.1 — Phân tích TARGET

```python
app = pd.read_csv('application_train.csv')

print(app['TARGET'].value_counts(normalize=True))
# Expected: ~91.9% class 0, ~8.1% class 1

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(5, 4))
ax.pie(app['TARGET'].value_counts(),
       labels=['No default (0)', 'Default (1)'],
       autopct='%1.1f%%', colors=['#5DCAA5', '#D85A30'])
ax.set_title('Target distribution')
plt.tight_layout()
```

**Câu hỏi**: Tỷ lệ imbalance là bao nhiêu?
→ Quyết định: dùng `class_weight='balanced'` hoặc SMOTE, và dùng ROC-AUC / PR-AUC thay vì Accuracy.

---

### 2.2 — Missing value analysis

```python
def missing_summary(df, name=''):
    miss = df.isnull().sum()
    miss_pct = miss / len(df) * 100
    result = pd.DataFrame({
        'missing_count': miss,
        'missing_pct': miss_pct,
        'dtype': df.dtypes
    }).query('missing_count > 0').sort_values('missing_pct', ascending=False)
    print(f"\n=== {name} — {len(result)} columns with missing values ===")
    return result

app_miss = missing_summary(app, 'application_train')
```

**Phân loại missing theo mức độ**:

```python
# Nhóm 1: > 40% missing → cân nhắc drop hoặc chỉ dùng binary flag "có/không có"
high_miss = app_miss[app_miss['missing_pct'] > 40]

# Nhóm 2: 10-40% missing → impute có chiến lược (median cho numeric, mode cho categorical)
mid_miss = app_miss[(app_miss['missing_pct'] > 10) & (app_miss['missing_pct'] <= 40)]

# Nhóm 3: < 10% missing → impute đơn giản
low_miss = app_miss[app_miss['missing_pct'] <= 10]

print(f"High missing (>40%): {len(high_miss)} cols")
print(f"Mid missing (10-40%): {len(mid_miss)} cols")
print(f"Low missing (<10%): {len(low_miss)} cols")
```

**Câu hỏi**: Các features quan trọng (EXT_SOURCE_1/2/3, AMT_*) có missing nhiều không?
→ EXT_SOURCE_1 thường missing ~56% → tạo thêm flag `EXT_SOURCE_1_missing` (binary) song song với impute.

---

### 2.3 — Phân tích features số theo TARGET

Mục tiêu: tìm features có **phân phối khác biệt rõ** giữa class 0 và class 1.

```python
numeric_cols = app.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['SK_ID_CURR', 'TARGET']]

# Tính mean theo TARGET cho từng feature
target_corr = app[numeric_cols + ['TARGET']].groupby('TARGET').mean().T
target_corr['ratio_1_vs_0'] = target_corr[1] / target_corr[0]
target_corr['diff'] = abs(target_corr[1] - target_corr[0])
target_corr = target_corr.sort_values('diff', ascending=False)

print("Top 20 features có phân phối khác biệt nhất giữa 2 class:")
print(target_corr.head(20))
```

**Visualize top features**:

```python
top_features = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1',
                'DAYS_BIRTH', 'AMT_CREDIT', 'AMT_ANNUITY',
                'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flatten(), top_features):
    for target_val, color, label in [(0, '#5DCAA5', 'No default'), (1, '#D85A30', 'Default')]:
        data = app[app['TARGET'] == target_val][col].dropna()
        ax.hist(data, bins=50, alpha=0.5, color=color, label=label, density=True)
    ax.set_title(col)
    ax.legend(fontsize=8)
plt.tight_layout()
```

**Câu hỏi**: Features nào có histogram 2 class tách biệt rõ?
→ Đây là features ưu tiên khi engineer (tạo bins, ratios từ chúng).

---

### 2.3b — Correlation heatmap (top features vs TARGET)

Sau khi đã xác định `top_features` từ bước trên, heatmap giúp trả lời 2 câu hỏi cùng lúc: feature nào tương quan với TARGET, và feature nào tương quan *với nhau* (→ candidates drop ở Bước 6.2).

> **Giới hạn top 20–30 features.** Heatmap 120+ features không đọc được.

```python
top20 = target_corr.head(20).index.tolist()
corr_data = app[top20 + ['TARGET']].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_data, dtype=bool))  # chỉ show lower triangle
sns.heatmap(
    corr_data,
    mask=mask,
    annot=True, fmt='.2f',
    cmap='RdYlGn', center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax
)
ax.set_title('Correlation heatmap — top 20 features + TARGET')
plt.tight_layout()
```

**Đọc heatmap:**

- Cột/hàng `TARGET`: features nào có |corr| cao → signal mạnh
- Các cặp features có |corr| > 0.85 → ghi lại, drop ở Bước 6.2

**Lưu ý quan trọng**: Pearson correlation chỉ bắt được **linear relationship**. Tree-based models (XGBoost, LightGBM) bắt được non-linear — nên không dùng heatmap này thay thế feature importance ở Bước 6.3. Dùng song song: heatmap tìm redundancy, feature importance tìm predictive power thực sự.

---

### 2.4 — Phân tích features categorical

```python
cat_cols = app.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    default_rate = app.groupby(col)['TARGET'].mean().sort_values(ascending=False)
    print(f"\n{col}:")
    print(default_rate.to_string())
```

**Visualize default rate theo category**:

```python
for col in cat_cols[:6]:  # top 6 categorical
    fig, ax = plt.subplots(figsize=(8, 4))
    rates = app.groupby(col)['TARGET'].mean().sort_values(ascending=False)
    counts = app[col].value_counts()
    
    bars = ax.bar(rates.index, rates.values, color='#5DCAA5', alpha=0.8)
    ax.axhline(app['TARGET'].mean(), color='#D85A30', linestyle='--', label='Overall mean')
    
    # Thêm count trên mỗi bar
    for bar, (cat, count) in zip(bars, counts.items()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'n={count:,}', ha='center', va='bottom', fontsize=8)
    
    ax.set_title(f'Default rate by {col}')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
```

**Câu hỏi**: Category nào có default rate cao bất thường?
→ Quyết định encoding: target encoding cho categories có nhiều levels, one-hot cho ít levels.

---

### 2.5 — Phát hiện anomalies

Một số cột trong dataset này có giá trị bất thường đã được biết:

```python
# DAYS_EMPLOYED: giá trị 365243 là placeholder cho "retired/unemployed"
print(app['DAYS_EMPLOYED'].value_counts().head())
print(f"\nRows với DAYS_EMPLOYED=365243: {(app['DAYS_EMPLOYED'] == 365243).sum():,}")

# Xử lý:
app['DAYS_EMPLOYED_ANOMALY'] = (app['DAYS_EMPLOYED'] == 365243).astype(int)
app['DAYS_EMPLOYED'] = app['DAYS_EMPLOYED'].replace(365243, np.nan)

# DAYS_* columns: giá trị âm (ngày trước ngày apply)
# DAYS_BIRTH: âm → tuổi thực = abs(DAYS_BIRTH) / 365
app['AGE_YEARS'] = abs(app['DAYS_BIRTH']) / 365

# Kiểm tra outliers bằng IQR
for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']:
    Q1, Q3 = app[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = ((app[col] < Q1 - 3*IQR) | (app[col] > Q3 + 3*IQR)).sum()
    print(f"{col}: {outliers} outliers")
```

---

## Bước 3 — EDA từng bảng phụ

### Template EDA cho bảng phụ

Áp dụng template này cho: `bureau`, `previous_application`, `pos_cash`, `installments`, `credit_card`.

```python
def eda_auxiliary_table(df, name, key_col, app_df):
    print(f"\n{'='*50}")
    print(f"TABLE: {name}")
    print(f"Shape: {df.shape}")
    print(f"Unique {key_col}: {df[key_col].nunique():,}")
    print(f"Avg rows per customer: {len(df)/df[key_col].nunique():.1f}")
    
    # Missing values
    miss_pct = df.isnull().mean() * 100
    print(f"\nColumns with >20% missing:")
    print(miss_pct[miss_pct > 20].sort_values(ascending=False))
    
    # Merge với TARGET để xem correlation
    if 'SK_ID_CURR' in df.columns:
        merged = df.merge(app_df[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR', how='left')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != key_col]
        
        correlations = merged[numeric_cols + ['TARGET']].corr()['TARGET'].abs()
        print(f"\nTop 10 features correlated with TARGET:")
        print(correlations.sort_values(ascending=False).head(11))
    
    return df
```

---

### 3.1 — bureau.csv

```python
bureau = pd.read_csv('bureau.csv')
eda_auxiliary_table(bureau, 'bureau', 'SK_ID_CURR', app)

# Phân tích CREDIT_ACTIVE (Active/Closed/Sold/Bad debt)
print("\nCREDIT_ACTIVE distribution:")
print(bureau['CREDIT_ACTIVE'].value_counts(normalize=True))

# Default rate by credit active status
active_target = bureau.merge(app[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR')
print("\nDefault rate by CREDIT_ACTIVE:")
print(active_target.groupby('CREDIT_ACTIVE')['TARGET'].mean().sort_values(ascending=False))

# AMT_CREDIT_SUM_DEBT distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT']):
    for t, color in [(0, '#5DCAA5'), (1, '#D85A30')]:
        data = active_target[active_target['TARGET']==t][col].clip(0, active_target[col].quantile(0.99))
        ax.hist(data, bins=50, alpha=0.5, color=color, density=True, label=f'TARGET={t}')
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
```

**Quyết định aggregation từ EDA bureau**:

| Insight từ EDA | Feature cần tạo |
|---|---|
| Khách hàng có nhiều khoản "Bad debt" → default cao | `bureau_BAD_DEBT_count` |
| Tỷ lệ AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM cao → rủi ro | `bureau_debt_credit_ratio` (aggregate mean per customer) |
| Số khoản đang active nhiều → overleverage | `bureau_ACTIVE_count`, `bureau_ACTIVE_ratio` |
| DAYS_CREDIT gần đây → đang vay nhiều | `bureau_DAYS_CREDIT_mean`, `bureau_DAYS_CREDIT_max` |

---

### 3.2 — bureau_balance.csv

```python
bureau_bal = pd.read_csv('bureau_balance.csv')

# STATUS encoding: C=closed, X=unknown, 0=no DPD, 1-5=months past due
print("STATUS distribution:")
print(bureau_bal['STATUS'].value_counts(normalize=True))

# Tạo binary: có bao giờ quá hạn không
bureau_bal['ever_late'] = bureau_bal['STATUS'].isin(['1','2','3','4','5']).astype(int)

# Aggregate về bureau level trước
bureau_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(
    months_count=('MONTHS_BALANCE', 'count'),
    ever_late=('ever_late', 'max'),
    max_dpd=('STATUS', lambda x: x[x.isin(['1','2','3','4','5'])].astype(float).max() if x.isin(['1','2','3','4','5']).any() else 0)
).reset_index()

# Sau đó join với bureau → aggregate tiếp về SK_ID_CURR
```

**Note**: Bureau_balance cần aggregate 2 cấp: `SK_ID_BUREAU` → `SK_ID_CURR`.

---

### 3.3 — previous_application.csv

```python
prev = pd.read_csv('previous_application.csv')
eda_auxiliary_table(prev, 'previous_application', 'SK_ID_CURR', app)

# Contract status distribution
print("\nNAME_CONTRACT_STATUS:")
print(prev['NAME_CONTRACT_STATUS'].value_counts(normalize=True))

# Default rate theo contract status
prev_target = prev.merge(app[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR')
print("\nDefault rate by contract status:")
print(prev_target.groupby('NAME_CONTRACT_STATUS')['TARGET'].mean())

# AMT_APPLICATION vs AMT_CREDIT: credit được approve khác application bao nhiêu?
prev['CREDIT_TO_APPLICATION_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION']
print("\nCredit-to-application ratio stats:")
print(prev['CREDIT_TO_APPLICATION_RATIO'].describe())

# DAYS_DECISION: đơn vay gần nhất vs xa nhất
prev_sorted = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False])
print("\nDAYS_DECISION range:", prev['DAYS_DECISION'].min(), "to", prev['DAYS_DECISION'].max())
```

**Quyết định time-sliced aggregation từ EDA**:

```python
# Insight từ EDA: đơn vay gần đây relevant hơn đơn vay cũ
# → Tạo aggregates cho last 1, 3, 5 applications

prev_sorted = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False])

for n in [1, 3, 5]:
    prev_last_n = prev_sorted.groupby('SK_ID_CURR').head(n)
    # Aggregate trên prev_last_n...
    print(f"Last {n} applications: {prev_last_n.shape}")
```

---

### 3.4 — installments_payments.csv

```python
inst = pd.read_csv('installments_payments.csv')

# KEY INSIGHT: chênh lệch giữa số phải trả và thực trả
inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT']
inst['DAYS_LATE'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['IS_LATE'] = (inst['DAYS_LATE'] > 0).astype(int)

# Distribution của PAYMENT_RATIO
print("PAYMENT_RATIO stats:")
print(inst['PAYMENT_RATIO'].describe())

# Merge với TARGET để confirm importance
inst_target = inst.merge(app[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR')
print("\nAvg PAYMENT_RATIO by TARGET:")
print(inst_target.groupby('TARGET')['PAYMENT_RATIO'].mean())
print("\nAvg IS_LATE by TARGET:")
print(inst_target.groupby('TARGET')['IS_LATE'].mean())
```

---

### 3.5 — credit_card_balance.csv

```python
cc = pd.read_csv('credit_card_balance.csv')

# Credit utilization: key metric trong credit scoring
cc['CREDIT_UTILIZATION'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']

cc_target = cc.merge(app[['SK_ID_CURR', 'TARGET']], on='SK_ID_CURR')
print("Credit utilization by TARGET:")
print(cc_target.groupby('TARGET')['CREDIT_UTILIZATION'].mean())

# Payment behavior
cc['MIN_PAYMENT_RATIO'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']
```

---

## Bước 4 — Aggregation Strategy

Sau EDA, xây dựng aggregation một cách có hệ thống. **Nguyên tắc**: chỉ aggregate những features mà EDA đã cho thấy có sự khác biệt giữa 2 class.

### Template aggregation function

```python
def aggregate_table(df, group_col, agg_dict, prefix):
    """
    df: bảng phụ
    group_col: thường là 'SK_ID_CURR'
    agg_dict: {'col': ['mean', 'max', 'min', 'sum', 'count']}
    prefix: prefix tên cột output
    """
    agg_df = df.groupby(group_col).agg(agg_dict)
    agg_df.columns = [f'{prefix}_{col}_{stat}' for col, stat in agg_df.columns]
    agg_df = agg_df.reset_index()
    return agg_df
```

### Bureau aggregation

```python
bureau_agg = aggregate_table(
    bureau,
    group_col='SK_ID_CURR',
    agg_dict={
        'DAYS_CREDIT':          ['mean', 'max', 'min'],
        'DAYS_CREDIT_ENDDATE':  ['mean', 'max'],
        'AMT_CREDIT_SUM':       ['mean', 'sum', 'max'],
        'AMT_CREDIT_SUM_DEBT':  ['mean', 'sum', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum'],
        'CNT_CREDIT_PROLONG':   ['sum', 'mean'],
    },
    prefix='bureau'
)

# Thêm count và ratio features
bureau_counts = bureau.groupby('SK_ID_CURR').agg(
    bureau_total_count=('SK_ID_BUREAU', 'count'),
    bureau_active_count=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
    bureau_closed_count=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
    bureau_bad_debt_count=('CREDIT_ACTIVE', lambda x: (x == 'Bad debt').sum()),
).reset_index()

bureau_counts['bureau_active_ratio'] = (
    bureau_counts['bureau_active_count'] / bureau_counts['bureau_total_count']
)

bureau_agg = bureau_agg.merge(bureau_counts, on='SK_ID_CURR', how='left')
```

### Previous application aggregation (với time-slicing)

```python
prev_sorted = prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False])

# Aggregate toàn bộ
prev_all_agg = aggregate_table(
    prev, 'SK_ID_CURR',
    {
        'AMT_APPLICATION': ['mean', 'max', 'sum'],
        'AMT_CREDIT':      ['mean', 'max', 'sum'],
        'AMT_ANNUITY':     ['mean', 'max'],
        'DAYS_DECISION':   ['mean', 'min'],
        'CREDIT_TO_APPLICATION_RATIO': ['mean', 'min', 'max'],
    },
    prefix='prev_all'
)

# Aggregate last 3
prev_last3_agg = aggregate_table(
    prev_sorted.groupby('SK_ID_CURR').head(3), 'SK_ID_CURR',
    {
        'AMT_APPLICATION': ['mean', 'max'],
        'AMT_CREDIT':      ['mean', 'max'],
        'CREDIT_TO_APPLICATION_RATIO': ['mean'],
    },
    prefix='prev_last3'
)

# Refused count (domain insight: nhiều lần bị từ chối → rủi ro)
prev_refused = prev.groupby('SK_ID_CURR').agg(
    prev_refused_count=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
    prev_approved_count=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
).reset_index()
prev_refused['prev_refused_ratio'] = (
    prev_refused['prev_refused_count'] /
    (prev_refused['prev_refused_count'] + prev_refused['prev_approved_count'])
)
```

### Installments aggregation

```python
inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT']
inst['DAYS_LATE'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
inst['IS_LATE'] = (inst['DAYS_LATE'] > 0).astype(int)

inst_agg = aggregate_table(
    inst, 'SK_ID_CURR',
    {
        'PAYMENT_RATIO': ['mean', 'min'],
        'PAYMENT_DIFF':  ['mean', 'sum'],
        'DAYS_LATE':     ['mean', 'max'],
        'IS_LATE':       ['mean', 'sum'],  # mean = late rate, sum = total late count
        'AMT_PAYMENT':   ['sum', 'mean'],
    },
    prefix='inst'
)

# Time-sliced: last 90, 180, 365 days
for days in [90, 180, 365]:
    inst_recent = inst[inst['DAYS_INSTALMENT'] > -days]
    inst_recent_agg = aggregate_table(
        inst_recent, 'SK_ID_CURR',
        {'PAYMENT_RATIO': ['mean'], 'IS_LATE': ['mean', 'sum']},
        prefix=f'inst_last{days}d'
    )
```

---

## Bước 5 — Engineered Features (Ratio & Interaction)

Đây là bước tạo features từ **domain knowledge**, không phải từ aggregation.

### 5.1 — Ratio features từ application_train

```python
# Credit burden
app['CREDIT_INCOME_RATIO']    = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
app['ANNUITY_INCOME_RATIO']   = app['AMT_ANNUITY'] / app['AMT_INCOME_TOTAL']
app['CREDIT_ANNUITY_RATIO']   = app['AMT_CREDIT'] / app['AMT_ANNUITY']  # loan term proxy

# Down payment
app['CREDIT_GOODS_RATIO']     = app['AMT_CREDIT'] / app['AMT_GOODS_PRICE']
app['DOWN_PAYMENT']           = app['AMT_GOODS_PRICE'] - app['AMT_CREDIT']
app['DOWN_PAYMENT_RATIO']     = app['DOWN_PAYMENT'] / app['AMT_GOODS_PRICE']

# Age & employment
app['AGE_YEARS']              = abs(app['DAYS_BIRTH']) / 365
app['EMPLOYED_YEARS']         = abs(app['DAYS_EMPLOYED']) / 365
app['EMPLOYMENT_AGE_RATIO']   = app['EMPLOYED_YEARS'] / app['AGE_YEARS']

# Income per family member
app['INCOME_PER_PERSON']      = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS'].fillna(1)
app['CHILDREN_RATIO']         = app['CNT_CHILDREN'] / app['CNT_FAM_MEMBERS'].fillna(1)

# EXT_SOURCE combinations (thường là top features)
app['EXT_SOURCE_MEAN']        = app[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
app['EXT_SOURCE_STD']         = app[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
app['EXT_SOURCE_PROD']        = (app['EXT_SOURCE_1'] * app['EXT_SOURCE_2'] * app['EXT_SOURCE_3'])

# EXT_SOURCE / credit features (từ winning solution)
app['EXT3_CREDIT_RATIO']      = app['EXT_SOURCE_3'] / (app['AMT_CREDIT'] + 1)
app['EXT3_ANNUITY_RATIO']     = app['EXT_SOURCE_3'] / (app['AMT_ANNUITY'] + 1)
```

### 5.2 — Missing value flags

```python
# Các cột có missing nhiều → tạo binary flag trước khi impute
high_missing_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_YEAR',
                     'OWN_CAR_AGE', 'DAYS_LAST_PHONE_CHANGE']

for col in high_missing_cols:
    if col in app.columns:
        app[f'{col}_MISSING'] = app[col].isnull().astype(int)
```

### 5.3 — Cross-table ratio features

```python
# Sau khi đã aggregate bureau về app level
app = app.merge(bureau_agg, on='SK_ID_CURR', how='left')

# Debt-to-income (combining bureau data với application)
app['BUREAU_DEBT_INCOME_RATIO'] = (
    app['bureau_AMT_CREDIT_SUM_DEBT_sum'] / app['AMT_INCOME_TOTAL']
)

# Tổng nợ bên ngoài so với khoản vay hiện tại
app['BUREAU_TO_CURRENT_CREDIT_RATIO'] = (
    app['bureau_AMT_CREDIT_SUM_sum'] / app['AMT_CREDIT']
)
```

---

## Bước 6 — Feature Selection

Sau khi tạo xong tất cả features, cần loại bớt trước khi đưa vào model.

### 6.1 — Loại features có variance thấp

```python
from sklearn.feature_selection import VarianceThreshold

numeric_df = app.select_dtypes(include=[np.number]).drop(columns=['SK_ID_CURR', 'TARGET'])

vt = VarianceThreshold(threshold=0.01)
vt.fit(numeric_df)

low_variance_cols = numeric_df.columns[~vt.get_support()].tolist()
print(f"Dropping {len(low_variance_cols)} low-variance features")
```

### 6.2 — Loại features có correlation cao (redundant)

```python
# Tính correlation matrix
corr_matrix = numeric_df.corr().abs()

# Tìm pairs có correlation > 0.95
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]

print(f"Dropping {len(high_corr_cols)} high-correlation features")
```

### 6.3 — Feature importance từ LightGBM cơ bản

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

X = app.drop(columns=['SK_ID_CURR', 'TARGET'])
y = app['TARGET']

# Quick model để lấy feature importance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

quick_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42
)
quick_model.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': quick_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 30 features by importance:")
print(importance_df.head(30))

# Giữ top N features
top_n = 200
selected_features = importance_df.head(top_n)['feature'].tolist()
```

### 6.4 — Visualize feature importance

```python
fig, ax = plt.subplots(figsize=(10, 12))
top_30 = importance_df.head(30)
ax.barh(top_30['feature'][::-1], top_30['importance'][::-1], color='#5DCAA5')
ax.set_xlabel('Feature importance (gain)')
ax.set_title('Top 30 features')
plt.tight_layout()
```

---

## Bước 7 — Checklist trước khi đưa vào Model

```
□ Đã xử lý DAYS_EMPLOYED anomaly (365243 → NaN + flag)?
□ Đã tạo missing flags cho EXT_SOURCE_1/3?
□ Tất cả bảng phụ đã được aggregate về SK_ID_CURR?
□ Đã tạo time-sliced aggregates (last 3/5) cho previous_application?
□ Đã tạo installment payment ratio và late flags?
□ Đã tạo các ratio features từ application_train (credit/income, annuity/income)?
□ Đã drop low-variance và high-correlation features?
□ Đã chạy quick LightGBM để lấy feature importance sơ bộ?
□ Final feature count nằm trong khoảng 150-250?
□ Không có data leakage (không dùng TARGET khi tính features)?
```

---

## Tóm tắt: EDA insight → Feature decision

| EDA Observation | Feature Engineering Decision |
|---|---|
| EXT_SOURCE_1 missing 56% | Tạo `EXT_SOURCE_1_MISSING` flag + impute median |
| EXT_SOURCE_* là top predictors | Tạo mean, std, product của 3 cột này |
| DAYS_EMPLOYED có giá trị 365243 | Replace bằng NaN, tạo `DAYS_EMPLOYED_ANOMALY` flag |
| Khách hàng có "Bad debt" trong bureau → default cao | `bureau_BAD_DEBT_count` |
| PAYMENT_RATIO trong installments khác biệt rõ | `inst_PAYMENT_RATIO_mean`, `inst_IS_LATE_mean` |
| AMT_CREDIT / AMT_INCOME cao → overleverage | `CREDIT_INCOME_RATIO` |
| Đơn vay gần đây relevant hơn cũ | Time-sliced agg: last 3, 5 applications |
| Nhiều lần bị Refused → pattern rủi ro | `prev_refused_count`, `prev_refused_ratio` |
| Credit utilization thẻ tín dụng | `cc_CREDIT_UTILIZATION_mean` |
| EXT_SOURCE_3 / AMT_CREDIT | `EXT3_CREDIT_RATIO` (từ winning solution) |

---

*Guide version 1.0 — designed for Home Credit Default Risk dataset*
