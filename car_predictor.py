# ================================================================
#   🚗 CAR PRICE PREDICTOR — MACHINE LEARNING PROJECT
#   Author  : Kumar Shivam  |  Internpe Task
#   Dataset : Quikr Car Dataset (quikr_car.csv)
#   Models  : Linear Regression, Ridge, Lasso, Random Forest,
#             Gradient Boosting, XGBoost
# ================================================================

# ══════════════════════════════════════════════════════════════
# CELL 1 — Install & Import Libraries
# ══════════════════════════════════════════════════════════════
# !pip install xgboost --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from xgboost import XGBRegressor

print("✅ All Libraries Imported!")
print("🚗 Car Price Predictor — Kumar Shivam | Internpe")


# ══════════════════════════════════════════════════════════════
# CELL 2 — Load Dataset
# ══════════════════════════════════════════════════════════════
# Load directly from Google Drive (File ID from your link)
file_id = "17wcOi_UoPklpR2R2a2OPv9RjOvCnLxWj"
url = f"https://drive.google.com/uc?id={file_id}&export=download"

try:
    df = pd.read_csv(url)
    print("✅ Dataset loaded from Google Drive!")
except:
    # Fallback URL
    df = pd.read_csv("quikr_car.csv")
    print("✅ Dataset loaded from local file!")

print(f"\n📊 Shape   : {df.shape}")
print(f"📋 Columns : {list(df.columns)}")
print(f"\n🔍 First 5 rows:")
df.head()


# ══════════════════════════════════════════════════════════════
# CELL 3 — Raw Data Exploration
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("       RAW DATASET OVERVIEW")
print("=" * 55)
print(df.info())
print("\n📈 Sample values per column:")
for col in df.columns:
    print(f"  {col:<15}: {df[col].unique()[:4]}")


# ══════════════════════════════════════════════════════════════
# CELL 4 — Data Cleaning  🧹
# ══════════════════════════════════════════════════════════════
dfc = df.copy()

# ── 1. Clean 'name' — keep only first 3 words (brand + model)
dfc['name'] = dfc['name'].str.split().str[:3].str.join(' ')

# ── 2. Clean 'company' — strip whitespace
dfc['company'] = dfc['company'].str.strip()

# ── 3. Clean 'year' — keep only numeric 4-digit years
dfc['year'] = pd.to_numeric(dfc['year'], errors='coerce')
dfc = dfc[dfc['year'].between(1990, 2024)]

# ── 4. Clean 'Price' — remove "Ask For Price", commas, convert to int
dfc = dfc[dfc['Price'] != 'Ask For Price']
dfc['Price'] = dfc['Price'].astype(str).str.replace(',', '', regex=False)
dfc['Price'] = pd.to_numeric(dfc['Price'], errors='coerce')

# ── 5. Clean 'kms_driven' — remove 'kms', commas
dfc['kms_driven'] = dfc['kms_driven'].astype(str).str.replace(',', '', regex=False)
dfc['kms_driven'] = dfc['kms_driven'].str.extract(r'(\d+)').astype(float)

# ── 6. Clean 'fuel_type' — drop nulls
dfc['fuel_type'] = dfc['fuel_type'].str.strip()

# ── 7. Drop rows with any nulls
dfc.dropna(inplace=True)

# ── 8. Remove extreme outliers (Price)
dfc = dfc[dfc['Price'].between(50000, 5000000)]
dfc = dfc[dfc['kms_driven'] < 500000]

# ── 9. Reset index
dfc.reset_index(drop=True, inplace=True)

print(f"✅ Data Cleaning Complete!")
print(f"   Raw rows   : {len(df)}")
print(f"   Clean rows : {len(dfc)}")
print(f"   Dropped    : {len(df) - len(dfc)} rows")
print(f"\n📊 Clean Data Preview:")
dfc.head()


# ══════════════════════════════════════════════════════════════
# CELL 5 — EDA: Price Distribution
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0D1117')
for ax in axes:
    ax.set_facecolor('#161B22')

# Raw price distribution
axes[0].hist(dfc['Price'], bins=60, color='#4FC3F7',
             edgecolor='none', alpha=0.85)
axes[0].set_title('💰 Car Price Distribution', color='white',
                  fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel('Price (₹)', color='#aaa', fontsize=11)
axes[0].set_ylabel('Count', color='#aaa', fontsize=11)
axes[0].tick_params(colors='white')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_color('#333')
axes[0].spines['left'].set_color('#333')
axes[0].grid(axis='y', alpha=0.2, color='white')

# Log-transformed price
axes[1].hist(np.log1p(dfc['Price']), bins=60, color='#81C784',
             edgecolor='none', alpha=0.85)
axes[1].set_title('📊 Log(Price) Distribution — Normalized',
                  color='white', fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel('log(Price)', color='#aaa', fontsize=11)
axes[1].set_ylabel('Count', color='#aaa', fontsize=11)
axes[1].tick_params(colors='white')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_color('#333')
axes[1].spines['left'].set_color('#333')
axes[1].grid(axis='y', alpha=0.2, color='white')

plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()


# ══════════════════════════════════════════════════════════════
# CELL 6 — EDA: Top Car Brands by Count & Avg Price
# ══════════════════════════════════════════════════════════════
top_brands     = dfc['company'].value_counts().head(10)
brand_avg_price = dfc.groupby('company')['Price'].median().sort_values(ascending=False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0D1117')
for ax in axes:
    ax.set_facecolor('#161B22')

grad_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_brands)))

# Most listed brands
bars = axes[0].barh(top_brands.index[::-1], top_brands.values[::-1],
                    color=grad_colors[::-1], edgecolor='none', height=0.65)
for bar, val in zip(bars, top_brands.values[::-1]):
    axes[0].text(val + 5, bar.get_y() + bar.get_height()/2,
                 str(val), va='center', color='white',
                 fontsize=11, fontweight='bold')
axes[0].set_title('🚗 Most Listed Car Brands', color='white',
                  fontsize=13, fontweight='bold', pad=10)
axes[0].tick_params(colors='white')
axes[0].set_xlabel('Number of Listings', color='#aaa', fontsize=11)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_color('#333')
axes[0].spines['left'].set_color('#333')
axes[0].grid(axis='x', alpha=0.2, color='white')

# Highest priced brands
grad_gold = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(brand_avg_price)))
bars2 = axes[1].barh(brand_avg_price.index[::-1],
                     brand_avg_price.values[::-1] / 1e5,
                     color=grad_gold[::-1], edgecolor='none', height=0.65)
for bar, val in zip(bars2, brand_avg_price.values[::-1] / 1e5):
    axes[1].text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f'₹{val:.1f}L', va='center', color='white',
                 fontsize=10, fontweight='bold')
axes[1].set_title('💰 Highest Median Price — Top Brands',
                  color='white', fontsize=13, fontweight='bold', pad=10)
axes[1].tick_params(colors='white')
axes[1].set_xlabel('Median Price (₹ Lakhs)', color='#aaa', fontsize=11)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_color('#333')
axes[1].spines['left'].set_color('#333')
axes[1].grid(axis='x', alpha=0.2, color='white')

plt.tight_layout()
plt.savefig('brand_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()


# ══════════════════════════════════════════════════════════════
# CELL 7 — EDA: Fuel Type & KMs Driven Analysis
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor('#0D1117')
for ax in axes:
    ax.set_facecolor('#161B22')

# Fuel type distribution
fuel_counts = dfc['fuel_type'].value_counts()
fuel_colors = ['#4FC3F7','#81C784','#FFD54F','#FF8A65','#CE93D8']
wedges, texts, autotexts = axes[0].pie(
    fuel_counts.values, labels=fuel_counts.index,
    colors=fuel_colors[:len(fuel_counts)],
    autopct='%1.1f%%', startangle=140,
    wedgeprops={'linewidth': 2, 'edgecolor': '#0D1117'})
for t in texts:     t.set_color('white'); t.set_fontsize(11)
for t in autotexts: t.set_color('white'); t.set_fontsize(10)
axes[0].set_title('⛽ Fuel Type Distribution',
                  color='white', fontsize=13, fontweight='bold', pad=10)

# Price by fuel type
fuel_price = dfc.groupby('fuel_type')['Price'].median().sort_values(ascending=False)
bars = axes[1].bar(fuel_price.index, fuel_price.values / 1e5,
                   color=fuel_colors[:len(fuel_price)], edgecolor='none', width=0.5)
for bar, val in zip(bars, fuel_price.values / 1e5):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f'₹{val:.1f}L', ha='center',
                 color='white', fontsize=10, fontweight='bold')
axes[1].set_title('💰 Median Price by Fuel Type',
                  color='white', fontsize=13, fontweight='bold', pad=10)
axes[1].tick_params(colors='white')
axes[1].set_ylabel('Price (₹ Lakhs)', color='#aaa')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_color('#333')
axes[1].spines['left'].set_color('#333')
axes[1].grid(axis='y', alpha=0.2, color='white')

# KMs Driven vs Price scatter
axes[2].scatter(dfc['kms_driven'] / 1000, dfc['Price'] / 1e5,
                alpha=0.3, s=15, color='#FF8A65', edgecolors='none')
axes[2].set_title('📍 KMs Driven vs Price',
                  color='white', fontsize=13, fontweight='bold', pad=10)
axes[2].set_xlabel('KMs Driven (thousands)', color='#aaa', fontsize=11)
axes[2].set_ylabel('Price (₹ Lakhs)', color='#aaa', fontsize=11)
axes[2].tick_params(colors='white')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['bottom'].set_color('#333')
axes[2].spines['left'].set_color('#333')
axes[2].grid(alpha=0.15, color='white')

plt.tight_layout()
plt.savefig('fuel_kms_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()


# ══════════════════════════════════════════════════════════════
# CELL 8 — EDA: Year vs Price Trend
# ══════════════════════════════════════════════════════════════
year_price = dfc.groupby('year')['Price'].median().reset_index()

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#161B22')

ax.fill_between(year_price['year'], year_price['Price'] / 1e5,
                alpha=0.25, color='#AB47BC')
ax.plot(year_price['year'], year_price['Price'] / 1e5,
        color='#CE93D8', lw=2.5, marker='o',
        markersize=7, markerfacecolor='white')
for x, y in zip(year_price['year'], year_price['Price'] / 1e5):
    ax.text(x, y + 0.15, f'₹{y:.1f}L', ha='center',
            color='white', fontsize=8, fontweight='bold')
ax.set_title('📅 Year vs Median Car Price Trend',
             color='white', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Manufacturing Year', color='#aaa', fontsize=11)
ax.set_ylabel('Median Price (₹ Lakhs)', color='#aaa', fontsize=11)
ax.tick_params(colors='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#333')
ax.spines['left'].set_color('#333')
ax.grid(alpha=0.2, color='white')

plt.tight_layout()
plt.savefig('year_price_trend.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()


# ══════════════════════════════════════════════════════════════
# CELL 9 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════
# Encode for correlation
dfc_corr = dfc.copy()
le = LabelEncoder()
for col in ['company', 'fuel_type', 'name']:
    dfc_corr[col] = le.fit_transform(dfc_corr[col].astype(str))

corr = dfc_corr[['company','year','kms_driven','fuel_type','Price']].corr()

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#161B22')
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, linecolor='#0D1117',
            ax=ax, annot_kws={'size': 13, 'weight': 'bold'})
ax.set_title('🔥 Feature Correlation Heatmap',
             color='white', fontsize=14, fontweight='bold', pad=12)
ax.tick_params(colors='white', labelsize=11)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()


# ══════════════════════════════════════════════════════════════
# CELL 10 — Feature Engineering & Preprocessing
# ══════════════════════════════════════════════════════════════
df_ml = dfc.copy()

# Encode categoricals
le_company  = LabelEncoder()
le_fuel     = LabelEncoder()
df_ml['company_enc']   = le_company.fit_transform(df_ml['company'])
df_ml['fuel_type_enc'] = le_fuel.fit_transform(df_ml['fuel_type'])

# Car age feature
df_ml['car_age'] = 2024 - df_ml['year']

# Log-transform target (reduces skew → better regression)
df_ml['log_price'] = np.log1p(df_ml['Price'])

features = ['company_enc', 'fuel_type_enc', 'car_age', 'kms_driven']
X = df_ml[features]
y = df_ml['log_price']   # predicting log price

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"✅ Preprocessing Done!")
print(f"   Train : {X_train.shape[0]} records")
print(f"   Test  : {X_test.shape[0]} records")
print(f"   Features : {features}")


# ══════════════════════════════════════════════════════════════
# CELL 11 — Train 6 Models
# ══════════════════════════════════════════════════════════════
models = {
    "Linear Regression"  : LinearRegression(),
    "Ridge Regression"   : Ridge(alpha=10.0),
    "Lasso Regression"   : Lasso(alpha=0.001),
    "Random Forest"      : RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, random_state=42),
    "XGBoost"            : XGBRegressor(n_estimators=200, learning_rate=0.08, verbosity=0, random_state=42),
}

results = {}
print(f"\n{'Model':<22} {'R² Score':>10} {'MAE (₹)':>12} {'RMSE (₹)':>12}")
print("─" * 60)
for name, model in models.items():
    use_scaled = name in ['Linear Regression','Ridge Regression','Lasso Regression']
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s  if use_scaled else X_test

    model.fit(Xtr, y_train)
    log_pred = model.predict(Xte)

    # Convert back from log space
    y_pred_actual  = np.expm1(log_pred)
    y_test_actual  = np.expm1(y_test)

    r2   = r2_score(y_test, log_pred)
    mae  = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

    results[name] = {
        'model'   : model,
        'r2'      : r2,
        'mae'     : mae,
        'rmse'    : rmse,
        'y_pred'  : y_pred_actual,
        'Xte'     : Xte,
        'scaled'  : use_scaled,
    }
    print(f"  {name:<21} {r2:>10.4f} {mae:>12,.0f} {rmse:>12,.0f}")


# ══════════════════════════════════════════════════════════════
# CELL 12 — Model Performance Dashboard 📊
# ══════════════════════════════════════════════════════════════
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor('#0D1117')
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

names  = list(results.keys())
short  = ['LR','Ridge','Lasso','RF','GB','XGB']
r2s    = [results[n]['r2']   for n in names]
maes   = [results[n]['mae']  / 1e4 for n in names]   # in 10k
rmses  = [results[n]['rmse'] / 1e4 for n in names]

colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7','#FF8C94']

def dark_bar(ax, xvals, yvals, title, ylabel, fmt='', color_list=None):
    ax.set_facecolor('#161B22')
    c = color_list or colors
    bars = ax.bar(xvals, yvals, color=c[:len(xvals)], edgecolor='none', width=0.55)
    for bar, val in zip(bars, yvals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(yvals)*0.01,
                f'{val:.2f}{fmt}', ha='center', color='white',
                fontsize=9, fontweight='bold')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, color='#aaa', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    ax.set_ylim(0, max(yvals) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.grid(axis='y', alpha=0.2, color='white')

ax1 = fig.add_subplot(gs[0, 0])
dark_bar(ax1, short, r2s, '🎯 R² Score (Higher = Better)', 'R² Score')

ax2 = fig.add_subplot(gs[0, 1])
dark_bar(ax2, short, maes, '📉 MAE (Lower = Better)',
         'MAE (₹ ×10,000)')

ax3 = fig.add_subplot(gs[0, 2])
dark_bar(ax3, short, rmses, '📉 RMSE (Lower = Better)',
         'RMSE (₹ ×10,000)')

# Best model: Actual vs Predicted
best_name = max(results, key=lambda n: results[n]['r2'])
best      = results[best_name]
y_test_actual = np.expm1(y_test)

ax4 = fig.add_subplot(gs[1, 0:2])
ax4.set_facecolor('#161B22')
ax4.scatter(y_test_actual / 1e5, best['y_pred'] / 1e5,
            alpha=0.35, s=12, color='#4FC3F7', edgecolors='none')
max_val = max(y_test_actual.max(), best['y_pred'].max()) / 1e5
ax4.plot([0, max_val], [0, max_val], 'w--', lw=1.5, label='Perfect Prediction')
ax4.set_title(f'🎯 Actual vs Predicted Price — {best_name}',
              color='white', fontsize=12, fontweight='bold', pad=8)
ax4.set_xlabel('Actual Price (₹ Lakhs)', color='#aaa', fontsize=11)
ax4.set_ylabel('Predicted Price (₹ Lakhs)', color='#aaa', fontsize=11)
ax4.tick_params(colors='white')
ax4.legend(labelcolor='white', framealpha=0.2, fontsize=10)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_color('#333')
ax4.spines['left'].set_color('#333')
ax4.grid(alpha=0.15, color='white')

# Residuals
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#161B22')
residuals = y_test_actual - best['y_pred']
ax5.hist(residuals / 1e5, bins=50, color='#FFEAA7', edgecolor='none', alpha=0.85)
ax5.axvline(0, color='white', lw=1.5, linestyle='--')
ax5.set_title('📊 Residuals Distribution', color='white',
              fontsize=12, fontweight='bold', pad=8)
ax5.set_xlabel('Error (₹ Lakhs)', color='#aaa', fontsize=11)
ax5.set_ylabel('Count', color='#aaa', fontsize=11)
ax5.tick_params(colors='white')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_color('#333')
ax5.spines['left'].set_color('#333')
ax5.grid(axis='y', alpha=0.2, color='white')

fig.suptitle('🚗 Car Price Predictor — Model Performance Dashboard',
             color='white', fontsize=16, fontweight='bold', y=1.01)
plt.savefig('model_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()

print(f"\n🏆 Best Model : {best_name}")
print(f"   R² Score   : {best['r2']:.4f}")
print(f"   MAE        : ₹{best['mae']:,.0f}")
print(f"   RMSE       : ₹{best['rmse']:,.0f}")


# ══════════════════════════════════════════════════════════════
# CELL 13 — Feature Importance (Random Forest) 🌲
# ══════════════════════════════════════════════════════════════
rf_model  = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_,
                      index=['Brand','Fuel Type','Car Age','KMs Driven']
                      ).sort_values(ascending=True)

feat_colors = ['#FFD700' if v == feat_imp.max() else '#4FC3F7'
               for v in feat_imp.values]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#161B22')
bars = ax.barh(feat_imp.index, feat_imp.values,
               color=feat_colors, edgecolor='none', height=0.5)
for bar, val in zip(bars, feat_imp.values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', color='white',
            fontsize=12, fontweight='bold')
ax.set_title('🌲 Feature Importance — What Drives Car Price?',
             color='white', fontsize=14, fontweight='bold', pad=12)
ax.tick_params(colors='white', labelsize=12)
ax.set_xlabel('Importance Score', color='#aaa', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#333')
ax.spines['left'].set_color('#333')
ax.grid(axis='x', alpha=0.2, color='white')
g_patch = mpatches.Patch(color='#FFD700', label='Most Important')
b_patch = mpatches.Patch(color='#4FC3F7', label='Others')
ax.legend(handles=[g_patch, b_patch], labelcolor='white',
          framealpha=0.2, fontsize=11)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.show()


# ══════════════════════════════════════════════════════════════
# CELL 14 — 🚗 LIVE CAR PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════
def predict_car_price(company, fuel_type, year, kms_driven):
    """
    Predict the resale price of a used car.

    Parameters:
    -----------
    company    : str  — e.g. 'Maruti', 'Hyundai', 'Honda'
    fuel_type  : str  — 'Petrol', 'Diesel', 'CNG', 'LPG'
    year       : int  — Manufacturing year e.g. 2018
    kms_driven : int  — Kilometres driven e.g. 45000
    """
    best_model = results[best_name]['model']

    try:
        comp_enc = le_company.transform([company])[0]
    except ValueError:
        comp_enc = 0   # default if unseen company
    try:
        fuel_enc = le_fuel.transform([fuel_type])[0]
    except ValueError:
        fuel_enc = 0

    car_age = 2024 - year
    inp     = np.array([[comp_enc, fuel_enc, car_age, kms_driven]])

    if results[best_name]['scaled']:
        inp = scaler.transform(inp)

    log_pred      = best_model.predict(inp)[0]
    predicted_price = np.expm1(log_pred)

    # Price range (±15%)
    low  = predicted_price * 0.85
    high = predicted_price * 1.15

    print("\n" + "🚗" * 22)
    print("      CAR PRICE PREDICTOR — KUMAR SHIVAM")
    print("🚗" * 22)
    print(f"\n  🏷️  Brand         : {company}")
    print(f"  ⛽ Fuel Type     : {fuel_type}")
    print(f"  📅 Year          : {year}  (Age: {car_age} years)")
    print(f"  🛣️  KMs Driven    : {kms_driven:,} km")
    print(f"\n  ─────────────────────────────────────")
    print(f"  🤖 Model Used    : {best_name}")
    print(f"  ─────────────────────────────────────")
    print(f"\n  💰 Estimated Price : ₹ {predicted_price:,.0f}")
    print(f"  📊 Price Range     : ₹ {low:,.0f}  –  ₹ {high:,.0f}")
    print(f"  💵 In Lakhs        : ₹ {predicted_price/1e5:.2f} Lakh")
    print("\n" + "🚗" * 22)
    return predicted_price

# ── Test Predictions ──────────────────────────────────────────
predict_car_price(
    company    = 'Maruti',
    fuel_type  = 'Petrol',
    year       = 2017,
    kms_driven = 45000
)

predict_car_price(
    company    = 'Hyundai',
    fuel_type  = 'Diesel',
    year       = 2019,
    kms_driven = 30000
)

predict_car_price(
    company    = 'Honda',
    fuel_type  = 'Petrol',
    year       = 2015,
    kms_driven = 80000
)


# ══════════════════════════════════════════════════════════════
# CELL 15 — Final Summary 🏁
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("   🚗 PROJECT SUMMARY — CAR PRICE PREDICTOR")
print("=" * 60)
print(f"  Dataset      : Quikr Used Car Listings")
print(f"  Clean Records: {len(dfc)}")
print(f"  Features     : Brand, Fuel Type, Car Age, KMs Driven")
print(f"  Models       : {len(models)}")
print()
for name, res in sorted(results.items(), key=lambda x: -x[1]['r2']):
    bar  = '█' * int(res['r2'] * 30)
    print(f"  {name:<22} R²={res['r2']:.4f}  {bar}")
print()
print(f"  🥇 Best Model  : {best_name}")
print(f"  🎯 R² Score    : {results[best_name]['r2']:.4f}")
print(f"  📉 MAE         : ₹{results[best_name]['mae']:,.0f}")
print(f"  📉 RMSE        : ₹{results[best_name]['rmse']:,.0f}")
print("=" * 60)
print("  Built with ❤️  by Kumar Shivam  |  Internpe Task")
print("=" * 60)