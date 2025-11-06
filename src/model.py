import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

#Tensorflow & Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#scikit learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Libraries Imported")

#Create Directories
# model_path = "../models"
# if not os.path.exists(model_path):
#     os.makedirs(model_path)
#     print("Model Path Created")

#set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#load Data
print("\n Loading Dataset..")
try:
    df = pd.read_csv("cleaned_data/customer_data_scaled.csv")
    print(f"Dataset Loaded")
except FileNotFoundError:
    df = pd.read_csv('../cleaned_data/customer_data_scaled.csv')
    print(f"Dataset Loaded from Parent Directory")
print(f"Shape: {df.shape}")
print(f"Columns:{len(df.columns)}")

# Prepare Data for modeling
print("Data Preparation")
try:
    df_raw = pd.read_csv('cleaned_data/customer_data_rfm.csv')
    y_actual = df_raw['moneary'].values
except:
    df_raw = pd.read_csv('../cleaned_data/customer_data_rfm.csv')
    y_actual = df_raw['monetary'].values

print(f"\n Target Variable: monetary(Total Spending)")
print(f"   Min: ${y_actual.min():.2f}")
print(f"   Max: ${y_actual.max():.2f}")
print(f"   Mean: ${y_actual.mean():.2f}")
print(f"   Std: ${y_actual.std():.2f}")

#Features: Drop non-numeric and target-realed columns
drop_cols = ['customer_id', 'registration_date', 'last_purchase_date', 
             'monetary', 'rfm_score', 'segment','product_category', 'gender']

X = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

print(f"\n Features selected: {len(X.columns)}")
print(f" Feature names: {list(X.columns)[:10]}")

#Target variable (scaled)
y = df['monetary'].values

print(f"\n X shape: {X.shape}")
print(f" y shape: {y.shape}")

# Train Test Split
print("Spliting Train Test Data")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

print(f"\n Training set: {X_train.shape[0]} samples")
print(f" Test set: {X_test.shape[0]} samples")
print(f"Train-test split: 80-20")

# Build Neural Network Model

print("Building Nueral Network")
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),

    #First hidden layer
    layers.Dense(128,activation='relu',name='dense_1'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Second hidden layer
    layers.Dense(64, activation='relu', name='dense_2'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Third hidden layer
    layers.Dense(32, activation='relu', name='dense_3'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Fourth hidden layer
    layers.Dense(16, activation='relu', name='dense_4'),
    layers.Dropout(0.1),
    
    # Output layer (regression - single neuron)
    layers.Dense(1, activation='linear', name='output')
])

#compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=['mae','mse']
)

print("\n📐 Model Architecture:")
model.summary()

#CallBacks
print("Setting Up Callbacks")
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr = 0.00001,
    verbose=1
)

print("✓ Early stopping enabled (patience: 15 epochs)")
print("✓ Learning rate reduction enabled")

#Train Model
print("Training Model")

history = model.fit(
    X_train,y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks = [early_stop,reduce_lr],
    verbose=1
)


print("\n✓ Training complete!")
print(f"✓ Total epochs trained: {len(history.history['loss'])}")

#Evaluate on Test set
print("Model Evaluation")

print("Evaluating on test set")

#make predictions
y_pred_train = model.predict(X_train,verbose=0)
y_pred_test = model.predict(X_test,verbose=0)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n📊 TRAINING SET METRICS:")
print(f"   MAE:  {train_mae:.4f}")
print(f"   RMSE: {train_rmse:.4f}")
print(f"   R²:   {train_r2:.4f}")

print(f"\n📊 TEST SET METRICS:")
print(f"   MAE:  {test_mae:.4f}")
print(f"   RMSE: {test_rmse:.4f}")
print(f"   R²:   {test_r2:.4f}")

#Save Model
print("Saving Model")

model_path = '../models/clv_model.h5'
model.save(model_path)
print(f'Model saved: {model_path}')

#save Scaler
print(X.shape[1])

scaler_path = '../models/model_scaler.pkl'
with open(scaler_path,'wb') as f:
    pickle.dump(X,f)
print(f"Scaler saved:{scaler_path}")

#Visualizations - Training History
print("Creating Visualizations")

fig, axes = plt.subplots(1,2,figsize=(16,5))
fig.patch.set_facecolor('white')


# Loss curve
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# MAE curve
axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2, color='green')
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='orange')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('Model MAE Over Epochs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../output/model_training.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Training curves saved: output/model_training.png")
plt.show()


#Visualization - predictions vs actual
fig,axes = plt.subplots(2,2, figsize=(16,12))
fig.patch.set_facecolor('white')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# 1. Training set: Actual vs Predicted
axes[0, 0].scatter(y_train, y_pred_train, alpha=0.5, s=30, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
               'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Spending ($)', fontsize=10)
axes[0, 0].set_ylabel('Predicted Spending ($)', fontsize=10)
axes[0, 0].set_title('Training Set: Actual vs Predicted', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Test set: Actual vs Predicted
axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5, s=30, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
               'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Spending ($)', fontsize=10)
axes[0, 1].set_ylabel('Predicted Spending ($)', fontsize=10)
axes[0, 1].set_title('Test Set: Actual vs Predicted', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Residuals (Test set)
residuals = y_test - y_pred_test.flatten()
axes[1, 0].scatter(y_pred_test, residuals, alpha=0.5, s=30, color='purple')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Spending ($)', fontsize=10)
axes[1, 0].set_ylabel('Residuals ($)', fontsize=10)
axes[1, 0].set_title('Residual Plot (Test Set)', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 4. Residuals distribution
axes[1, 1].hist(residuals, bins=30, color='orange', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Residuals ($)', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)



plt.tight_layout()
plt.savefig('../output/model_predictions.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Predictions visualization saved: output/model_predictions.png")
plt.show()

print("Saving Metrics Report")


metrics_report = f"""
CLV DEEP LEARNING MODEL - PERFORMANCE REPORT
{'='*60}

Model Architecture:
- Input Features: {input_dim}
- Layers: 4 Dense + Dropout + BatchNorm
- Neurons: 128 -> 64 -> 32 -> 16 -> 1
- Activation: ReLU (hidden), Linear (output)
- Optimizer: Adam (lr=0.001)
- Loss: MSE

Training Configuration:
- Total Epochs: {len(history.history['loss'])}
- Batch Size: 32
- Train-Val Split: 80-20
- Callbacks: EarlyStopping, ReduceLROnPlateau

Dataset:
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Total Samples: {len(X_train) + len(X_test)}

TRAINING SET METRICS:
- Mean Absolute Error (MAE): {train_mae:.4f}
- Root Mean Squared Error (RMSE): {train_rmse:.4f}
- R² Score: {train_r2:.4f}

TEST SET METRICS:
- Mean Absolute Error (MAE): {test_mae:.4f}
- Root Mean Squared Error (RMSE): {test_rmse:.4f}
- R² Score: {test_r2:.4f}

Model Interpretation:
- On average, predictions are off by ${test_mae:.2f} (MAE)
- R² Score of {test_r2:.4f} means model explains {test_r2*100:.2f}% variance
- Model is {'well-tuned' if abs(train_r2 - test_r2) < 0.05 else 'needs tuning'}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
metrics_path = '../output/model_metrics.txt'
with open(metrics_path,'w') as f:
    f.write(metrics_report)

print(f"Metrics report saved: {metrics_path}")

