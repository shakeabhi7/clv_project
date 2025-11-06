import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (15, 12)

current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Try to load from data folder
try:
    csv_path = 'data/customers_data.csv'
    df = pd.read_csv(csv_path)
    print(f"✓ Successfully loaded from: {csv_path}")
except FileNotFoundError:
    # If not found, try parent directory
    csv_path = '../data/customers_data.csv'
    df = pd.read_csv(csv_path)
    print(f"✓ Successfully loaded from: {csv_path}")

print(f"✓ Loaded {len(df)} records")
print(f"✓ Shape: {df.shape}")

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📋 Column Names & Types:")
print(df.dtypes)

print(f"\n⚠️  NULL VALUES (Before Cleaning):")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])
print(f"Total NULL values: {df.isnull().sum().sum()}")

print(f"\n⚠️  DUPLICATES (Before Cleaning):")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate by customer_id: {df.duplicated(subset=['customer_id']).sum()}")

print(f"\n⚠️  BASIC STATISTICS (with issues):")
print(df.describe())

print(f"\n🎯 First 10 rows (may have issues):")
print(df.head(10))

# ============================================================================
# STEP 3: DATA CLEANING
# ============================================================================
print("\n" + "=" * 80)
print("CLEANING DATA")
print("=" * 80)

df_clean = df.copy()

# 1. Remove complete duplicates
print(f"\n🧹 Removing duplicates...")
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['customer_id'], keep='first')
removed_dups = initial_rows - len(df_clean)
print(f"   ✓ Removed {removed_dups} duplicate records")

# 2. Remove rows with NULL values
print(f"\n🧹 Removing NULL values...")
initial_rows = len(df_clean)
df_clean = df_clean.dropna()
removed_nulls = initial_rows - len(df_clean)
print(f"   ✓ Removed {removed_nulls} rows with NULL values")

# 3. Remove rows with negative values (data errors)
print(f"\n🧹 Removing rows with negative values...")
initial_rows = len(df_clean)
df_clean = df_clean[(df_clean['purchase_frequency'] > 0) & 
                     (df_clean['avg_order_value'] > 0) & 
                     (df_clean['total_spending'] > 0) &
                     (df_clean['num_orders'] > 0) &
                     (df_clean['customer_lifetime_days'] > 0)]
removed_negatives = initial_rows - len(df_clean)
print(f"   ✓ Removed {removed_negatives} rows with negative values")

# 4. Remove outliers (extreme values)
print(f"\n🧹 Removing outliers using IQR method...")
initial_rows = len(df_clean)

for col in ['avg_order_value', 'total_spending', 'num_orders']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

removed_outliers = initial_rows - len(df_clean)
print(f"   ✓ Removed {removed_outliers} outlier records")

# 5. Remove rows where last_purchase_date is after registration_date (logic error)
print(f"\n🧹 Fixing date inconsistencies...")
initial_rows = len(df_clean)
df_clean['registration_date'] = pd.to_datetime(df_clean['registration_date'])
df_clean['last_purchase_date'] = pd.to_datetime(df_clean['last_purchase_date'])
df_clean = df_clean[df_clean['last_purchase_date'] >= df_clean['registration_date']]
removed_dates = initial_rows - len(df_clean)
print(f"   ✓ Removed {removed_dates} rows with date inconsistencies")

# 6. Reset index
df_clean = df_clean.reset_index(drop=True)

# ============================================================================
# STEP 4: AFTER CLEANING - DATA QUALITY
# ============================================================================
print("\n" + "=" * 80)
print("AFTER CLEANING - CLEAN DATA")
print("=" * 80)

print(f"\n✅ Dataset Shape: {df_clean.shape}")
print(f"✅ Total rows remaining: {len(df_clean)}")
print(f"✅ NULL values: {df_clean.isnull().sum().sum()}")
print(f"✅ Duplicate rows: {df_clean.duplicated().sum()}")

print(f"\n📊 CLEAN DATA STATISTICS:")
print(df_clean.describe())

print(f"\n🎯 Sample of cleaned data (First 10 rows):")
print(df_clean.head(10))

# ============================================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print(f"\n👥 CUSTOMER DEMOGRAPHICS:")
print(f"   Total customers: {len(df_clean)}")
print(f"   Average age: {df_clean['age'].mean():.2f} years")
print(f"   Age range: {df_clean['age'].min()} - {df_clean['age'].max()} years")
print(f"\n   Gender distribution:")
print(df_clean['gender'].value_counts())

print(f"\n💰 SPENDING ANALYSIS:")
print(f"   Average total spending: ${df_clean['total_spending'].mean():.2f}")
print(f"   Median total spending: ${df_clean['total_spending'].median():.2f}")
print(f"   Total spending range: ${df_clean['total_spending'].min():.2f} - ${df_clean['total_spending'].max():.2f}")
print(f"   Average order value: ${df_clean['avg_order_value'].mean():.2f}")

print(f"\n📈 PURCHASE BEHAVIOR:")
print(f"   Average purchase frequency: {df_clean['purchase_frequency'].mean():.2f}")
print(f"   Average number of orders: {df_clean['num_orders'].mean():.2f}")
print(f"   Average customer lifetime (days): {df_clean['customer_lifetime_days'].mean():.2f} days")

print(f"\n🛍️  PRODUCT CATEGORY DISTRIBUTION:")
print(df_clean['product_category'].value_counts())

# ============================================================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 14))

# 1. Age distribution
ax1 = plt.subplot(3, 3, 1)
plt.hist(df_clean['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Age', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Distribution of Customer Age', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 2. Total spending distribution
ax2 = plt.subplot(3, 3, 2)
plt.hist(df_clean['total_spending'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
plt.xlabel('Total Spending ($)', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Distribution of Total Spending', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 3. Gender distribution
ax3 = plt.subplot(3, 3, 3)
gender_counts = df_clean['gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], edgecolor='black')
plt.xlabel('Gender', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Customer Gender Distribution', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 4. Purchase frequency distribution
ax4 = plt.subplot(3, 3, 4)
plt.hist(df_clean['purchase_frequency'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
plt.xlabel('Purchase Frequency', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Distribution of Purchase Frequency', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 5. Average order value distribution
ax5 = plt.subplot(3, 3, 5)
plt.hist(df_clean['avg_order_value'], bins=30, color='gold', edgecolor='black', alpha=0.7)
plt.xlabel('Average Order Value ($)', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Distribution of Average Order Value', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 6. Product category distribution
ax6 = plt.subplot(3, 3, 6)
category_counts = df_clean['product_category'].value_counts()
plt.barh(category_counts.index, category_counts.values, color='plum', edgecolor='black')
plt.xlabel('Count', fontsize=10)
plt.title('Product Category Distribution', fontsize=12, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# 7. Scatter plot: Age vs Total Spending
ax7 = plt.subplot(3, 3, 7)
plt.scatter(df_clean['age'], df_clean['total_spending'], alpha=0.5, s=30, color='teal')
plt.xlabel('Age', fontsize=10)
plt.ylabel('Total Spending ($)', fontsize=10)
plt.title('Age vs Total Spending', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

# 8. Scatter plot: Purchase Frequency vs Total Spending
ax8 = plt.subplot(3, 3, 8)
plt.scatter(df_clean['purchase_frequency'], df_clean['total_spending'], alpha=0.5, s=30, color='orange')
plt.xlabel('Purchase Frequency', fontsize=10)
plt.ylabel('Total Spending ($)', fontsize=10)
plt.title('Purchase Frequency vs Total Spending', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

# 9. Box plot: Spending by Gender
ax9 = plt.subplot(3, 3, 9)
df_clean.boxplot(column='total_spending', by='gender', ax=ax9)
plt.xlabel('Gender', fontsize=10)
plt.ylabel('Total Spending ($)', fontsize=10)
plt.title('Total Spending by Gender', fontsize=12, fontweight='bold')
plt.suptitle('')  # Remove automatic title

plt.tight_layout()
plt.savefig('data/eda_visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved to: data/eda_visualizations.png")
plt.show()

# ============================================================================
# STEP 7: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Select numeric columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
correlation_matrix = df_clean[numeric_cols].corr()

print(f"\n📊 Correlation Matrix:")
print(correlation_matrix)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Correlation heatmap saved to: data/correlation_heatmap.png")
plt.show()

# ============================================================================
# STEP 8: KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS FROM EDA")
print("=" * 80)

print(f"\n🔍 Top insights:")
print(f"   1. Total customers analyzed: {len(df_clean)}")
print(f"   2. Data quality issues removed: {len(df) - len(df_clean)}")
print(f"   3. Strongest correlation with total spending: {correlation_matrix['total_spending'].drop('total_spending').abs().idxmax()} ({correlation_matrix['total_spending'].drop('total_spending').abs().max():.3f})")
print(f"   4. Average customer lifetime value: ${df_clean['total_spending'].mean():.2f}")
print(f"   5. High-value customers (top 20%): ${df_clean['total_spending'].quantile(0.8):.2f} and above")

# ============================================================================
# STEP 9: SAVE CLEANED DATASET
# ============================================================================
print("\n" + "=" * 80)
print("SAVING CLEANED DATASET")
print("=" * 80)

cleaned_csv_path = 'data/customers_data_cleaned.csv'
df_clean.to_csv(cleaned_csv_path, index=False)

print(f"\n✓ Cleaned dataset saved to: {cleaned_csv_path}")
print(f"✓ Records: {len(df_clean)}")
print(f"✓ Columns: {len(df_clean.columns)}")

# ============================================================================
# STEP 10: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DAY 1 PART 2 - COMPLETE SUMMARY")
print("=" * 80)

print(f"""
📊 Data Quality Report:
   - Raw records: {len(df)}
   - Cleaned records: {len(df_clean)}
   - Records removed: {len(df) - len(df_clean)} ({((len(df) - len(df_clean)) / len(df) * 100):.1f}%)
   - NULL values removed: {removed_nulls}
   - Duplicates removed: {removed_dups}
   - Outliers removed: {removed_outliers}
   - Date inconsistencies fixed: {removed_dates}

📈 Key Statistics:
   - Customers: {len(df_clean)}
   - Average spending: ${df_clean['total_spending'].mean():.2f}
   - Average orders: {df_clean['num_orders'].mean():.2f}
   - Average age: {df_clean['age'].mean():.2f} years

📁 Output Files:
   - Cleaned data: data/customers_data_cleaned.csv
   - Visualizations: data/eda_visualizations.png
   - Correlation: data/correlation_heatmap.png

✅ Next Step: Day 2 - Feature Engineering + RFM Analysis
""")

print("=" * 80)