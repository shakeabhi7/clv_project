import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Model Optimization + Comparison
print("Model Optimization and Comparison")

np.random.seed(42)
tf.random.set_seed(42)

#Load Data
print('Dataset Loading')

try:
    df = pd.read_csv('cleaned_data/customer_data_scaled.csv')
except FileNotFoundError:
    df = pd.read_csv('../cleaned_data/customer_data_scaled.csv')

try:
    df_raw = pd.read_csv('cleaned_data/customer_data_rfm.csv')
except FileNotFoundError:
    df_raw = pd.read_csv('../cleaned_data/customer_data_rfm.csv')

print(f"Data loaded: {len(df)} records")
print(f"RFM Data Loaded: {len(df)} records")

# Prepare Data
print("Preparing Data")

drop_cols = ['customer_id', 'registration_date', 'last_purchase_date', 
             'monetary', 'rfm_score', 'segment','product_category', 'gender']

X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

y = df['monetary'].values


print(f"✓ Features: {X.shape[1]}")
print(f"✓ Samples: {len(X)}")

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")


#Load Pre trained Deep Leearning Model
print("Loading Pre-Trained Model")

try:
    model_dl = keras.models.load_model('../models/clv_model.h5',compile=False)
    print("Deep Learning Model Loaded")
except FileNotFoundError:
    print("Deep Learning model not found")

if model_dl:
    y_pred_dl = model_dl.predict(X_test,verbose=0)

    dl_mae = mean_absolute_error(y_test,y_pred_dl)
    dl_rmse = np.sqrt(mean_squared_error(y_test,y_pred_dl))
    dl_r2 = r2_score(y_test,y_pred_dl)

    print(f"✓ Deep Learning Model Loaded")
    print(f"   MAE: {dl_mae:.4f}")
    print(f"   RMSE: {dl_rmse:.4f}")
    print(f"   R²: {dl_r2:.4f}")


#Training Machine Learning Models
print("Training Traditional ML Models")
models_dict = {}

print("Training Random Forest..")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

rf_mae = mean_absolute_error(y_test,y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test,y_pred_rf))
rf_r2 = r2_score(y_test,y_pred_rf)


models_dict['Random Forest'] = {'model': model_rf, 'predictions': y_pred_rf,
                                'mae': rf_mae, 'rmse': rf_rmse, 'r2': rf_r2}
print(f" Random Forest: R² = {rf_r2:.4f}, MAE = {rf_mae:.4f}")

#Gradient Boosting
print("\n Training Gradient Boosting..")
model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

gb_mae = mean_absolute_error(y_test,y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test,y_pred_gb))
gb_r2 = r2_score(y_test,y_pred_gb)


models_dict['Gradient Boosting'] = {'model': model_gb, 'predictions': y_pred_gb,
                                    'mae': gb_mae, 'rmse': gb_rmse, 'r2': gb_r2}
print(f" Gradient Boosting: R² = {gb_r2:.4f}, MAE = {gb_mae:.4f}")

# 4. XGBoost
print("\n Training XGBoost...")
model_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
model_xgb.fit(X_train, y_train, verbose=False)
y_pred_xgb = model_xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)

models_dict['XGBoost'] = {'model': model_xgb, 'predictions': y_pred_xgb,
                         'mae': xgb_mae, 'rmse': xgb_rmse, 'r2': xgb_r2}
print(f" XGBoost: R² = {xgb_r2:.4f}, MAE = {xgb_mae:.4f}")

#Model Comparison
print("Model Comparison Results")

comparison_data = {
    'Model': ['Deep Learning', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'MAE': [dl_mae, rf_mae, gb_mae, xgb_mae],
    'RMSE': [dl_rmse, rf_rmse, gb_rmse, xgb_rmse],
    'R²': [dl_r2, rf_r2, gb_r2, xgb_r2]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('R²',ascending=False)

print("\n Model Performance Comparison")

#Find best model
best_model_name = comparison_df.iloc[0]['Model']
best_r2 = comparison_df.iloc[0]['R²']
best_mae = comparison_df.iloc[0]['MAE']

print(f"\n BEST MODEL: {best_model_name}")
print(f"   R² = {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"   MAE = {best_mae:.4f}")

# Save Best Model
print("Saving Best Model")
if best_model_name == "Deep Learning":
    print("Deep Learning model already")
else:
    best_model = models_dict[best_model_name]['model']
    with open('../models/clv_best_model.pkl','wb') as f:
        pickle.dump(best_model,f)
    print(f"Saved the Best model: models/clv_best_model.pkl ({best_model_name})")

# Visualizations - Model Comparison
print("Creating visualizations")

fig, axes = plt.subplots(2,2,figsize=(18,14))
fig.patch.set_facecolor('white')

#R2 score comparison
ax1= axes[0,0]
colors = ['gold' if model == best_model_name else 'skyblue' for model in comparison_df['Model']]
bars1 = ax1.barh(comparison_df['Model'], comparison_df['R²'], color=colors, edgecolor='black')
ax1.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax1.set_title('Model Comparison - R² Score', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 1)

for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width,bar.get_y() + bar.get_height()/2, f'{width:.4f}',
             ha='left',va='center',fontsize=10,fontweight='bold')
ax1.grid(axis='x',alpha=0.3)

#MAE Comparison
ax2 = axes[0, 1]
bars2 = ax2.barh(comparison_df['Model'], comparison_df['MAE'], color=colors, edgecolor='black')
ax2.set_xlabel('Mean Absolute Error', fontsize=11, fontweight='bold')
ax2.set_title('Model Comparison - MAE', fontsize=13, fontweight='bold')
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
            ha='left', va='center', fontsize=10, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. RMSE Comparison
ax3 = axes[1, 0]
bars3 = ax3.barh(comparison_df['Model'], comparison_df['RMSE'], color=colors, edgecolor='black')
ax3.set_xlabel('Root Mean Squared Error', fontsize=11, fontweight='bold')
ax3.set_title('Model Comparison - RMSE', fontsize=13, fontweight='bold')
for i, bar in enumerate(bars3):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
            ha='left', va='center', fontsize=10, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

#Overall performace
ax4 = axes[1,1]
x=np.arange(len(comparison_df))
width=0.25

#Normalize metrics for comparison
mae_norm = 1 - (comparison_df['MAE']/comparison_df['MAE'].max())
rmse_norm = 1 - (comparison_df['RMSE']/comparison_df['RMSE'].max())


bars_r2 = ax4.bar(x - width, comparison_df['R²'], width, label='R² Score', color='skyblue', edgecolor='black')
bars_mae = ax4.bar(x, mae_norm, width, label='MAE (normalized)', color='lightcoral', edgecolor='black')
bars_rmse = ax4.bar(x + width, rmse_norm, width, label='RMSE (normalized)', color='lightgreen', edgecolor='black')

ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('Overall Model Performance', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=9)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout(pad=6.0)
plt.savefig('../output/model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print(" Model comparison visualization saved")
plt.show()

#Best Model Analysis
figfig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('white')


if best_model_name == 'Deep Learning':
    best_pred = y_pred_dl
else:
    best_pred = models_dict[best_model_name]['predictions']

# Actual vs Predicted
axes[0].scatter(y_test, best_pred, alpha=0.5, s=40, color='teal')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Spending ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Predicted Spending ($)', fontsize=11, fontweight='bold')
axes[0].set_title(f'Best Model: {best_model_name}\nActual vs Predicted', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Residuals
residuals = y_test - best_pred.flatten()
axes[1].hist(residuals, bins=30, color='orange', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].set_title(f'Best Model: {best_model_name}\nResidual Distribution', fontsize=13, fontweight='bold')
axes[1].axvline(x=0, color='red', linestyle='--', lw=2)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../output/best_model_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print(" Best model analysis saved")
plt.show()

#Save Comparison Report
print("Saving Comparison Report")

comparison_report = f"""
CLV MODEL COMPARISON REPORT
{'='*70}

MODELS COMPARED:
1. Deep Learning Neural Network
2. Random Forest
3. Gradient Boosting
4. XGBoost

{'='*70}
PERFORMANCE COMPARISON:
{'='*70}

{comparison_df.to_string(index=False)}

{'='*70}
BEST MODEL: {best_model_name}
{'='*70}
R² Score:     {best_r2:.4f} ({best_r2*100:.2f}%)
MAE:          {best_mae:.4f}
RMSE:         {comparison_df.iloc[0]['RMSE']:.4f}

Model Type:   {'Deep Learning' if best_model_name == 'Deep Learning' else 'Traditional ML'}
Status:        Ready for Production

{'='*70}
DETAILED METRICS:
{'='*70}

DEEP LEARNING:
R²:   {dl_r2:.4f}
MAE:  {dl_mae:.4f}
RMSE: {dl_rmse:.4f}

RANDOM FOREST:
R²:   {rf_r2:.4f}
MAE:  {rf_mae:.4f}
RMSE: {rf_rmse:.4f}

GRADIENT BOOSTING:
R²:   {gb_r2:.4f}
MAE:  {gb_mae:.4f}
RMSE: {gb_rmse:.4f}

XGBOOST:
R²:   {xgb_r2:.4f}
MAE:  {xgb_mae:.4f}
RMSE: {xgb_rmse:.4f}

{'='*70}
RECOMMENDATION:
Use {best_model_name} for production deployment.
Model achieves {best_r2*100:.2f}% accuracy on test set.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('../output/model_comparison_report.txt', 'w') as f:
    f.write(comparison_report)

print("Comparison Report Saved: output/model_comparison.txt")


# Save comparison CSV
comparison_df.to_csv('../cleaned_data/model_comparison.csv', index=False)
print("Comparison CSV saved: output/model_comparison.csv")
