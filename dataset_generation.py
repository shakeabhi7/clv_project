import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================================
# This script generates realistic e-commerce data with quality issues
# (NULL values, duplicates, outliers) - just like real-world data
# ============================================================================

# Set random seed for reproducibility
np.random.seed(40)

print("=" * 80)
print("REALISTIC E-COMMERCE DATASET GENERATION")
print("(With real-world data quality issues)")
print("=" * 80)

# ============================================================================
# STEP 1: CREATE DATA FOLDER IF NOT EXISTS
# ============================================================================
if not os.path.exists('data'):
    os.makedirs('data')
    print("\nCreated 'data' folder")

# ============================================================================
# STEP 2: GENERATE BASE SYNTHETIC CUSTOMER DATA
# ============================================================================
print("\nGenerating base customer data...")

n_customers = 5000
base_date = datetime(2020, 1, 1)

customers = {
    'customer_id': [f'CUST_{i:05d}' for i in range(1, n_customers + 1)],
    'age': np.random.randint(18, 75, n_customers),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_customers),
    'registration_date': [base_date + timedelta(days=int(x)) 
                          for x in np.random.randint(0, 1400, n_customers)],
    'purchase_frequency': np.random.randint(1, 50, n_customers),
    'avg_order_value': np.random.uniform(5, 1000, n_customers),
    'total_spending': np.random.uniform(20, 15000, n_customers),
    'last_purchase_date': [base_date + timedelta(days=int(x)) 
                           for x in np.random.randint(0, 1460, n_customers)],
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'], 
                                         n_customers),
    'num_orders': np.random.randint(1, 150, n_customers),
    'customer_lifetime_days': np.random.randint(1, 1400, n_customers),
}

df = pd.DataFrame(customers)
print(f"✓ Generated {len(df)} initial records")

# ============================================================================
# STEP 3: ADD REALISTIC DATA QUALITY ISSUES
# ============================================================================
print("\nAdding real-world data quality issues...")

# 1. Add NULL values in random columns
print("   - Adding NULL values...")
null_prob = 0.03  # 3% null values

for col in ['age', 'gender', 'last_purchase_date', 'num_orders']:
    null_indices = np.random.choice(df.index, size=int(null_prob * len(df)), replace=False)
    df.loc[null_indices, col] = np.nan

print(f"     NULL values added")

# 2. Add duplicate records (realistic - same customer multiple times)
print("   - Adding duplicate records...")
n_duplicates = int(0.08 * len(df))  # 8% duplicates
duplicates = df.sample(n=n_duplicates)
df = pd.concat([df, duplicates], ignore_index=True)
print(f"     {n_duplicates} duplicate records added")

# 3. Add some unrealistic outliers (data entry errors)
print("   - Adding outliers (data entry errors)...")
outlier_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
df.loc[outlier_indices, 'avg_order_value'] = np.random.uniform(5000, 50000, len(outlier_indices))
print(f"     ✓ Outliers added")

# 4. Add some negative values (data errors)
print("   - Adding negative values (data errors)...")
error_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
df.loc[error_indices, 'purchase_frequency'] = -df.loc[error_indices, 'purchase_frequency']
print(f"     ✓ Negative values added")

# 5. Add inconsistent date formats (last_purchase_date after registration_date)
print("   - Adding date inconsistencies...")
for idx in np.random.choice(df.index, size=int(0.05 * len(df)), replace=False):
    if pd.notna(df.loc[idx, 'last_purchase_date']):
        df.loc[idx, 'last_purchase_date'] = df.loc[idx, 'registration_date'] - timedelta(days=100)
print(f"     ✓ Date inconsistencies added")

# ============================================================================
# STEP 4: DATASET OVERVIEW (BEFORE CLEANING)
# ============================================================================
print("\nRAW DATASET OVERVIEW (WITH ISSUES):")
print(f"   Total records: {len(df)}")
print(f"   Total columns: {len(df.columns)}")
print(f"\n   NULL values per column:")
print(df.isnull().sum())
print(f"\n   Total NULL values: {df.isnull().sum().sum()}")
print(f"   Duplicate rows: {df.duplicated().sum()}")

print("\nData Types:")
print(df.dtypes)

print("\nSample Data (First 10 rows - may contain issues):")
print(df.head(10))

print("\nBasic Statistics (with issues):")
print(df.describe())

# ============================================================================
# STEP 5: SAVE RAW DATASET TO CSV
# ============================================================================
print("\nSAVING RAW DATASET...")

csv_path = 'data/customers_data.csv'
df.to_csv(csv_path, index=False)

print(f"✓ Raw dataset saved to: {csv_path}")
print(f"✓ File size: {os.path.getsize(csv_path) / 1024:.2f} KB")

# ============================================================================
# STEP 6: VERIFY CSV FILE
# ============================================================================
print("\nVERIFICATION:")

df_verify = pd.read_csv(csv_path)
print(f"   Records in CSV: {len(df_verify)}")
print(f"   Columns in CSV: {list(df_verify.columns)}")
print(f"   Total NULL values: {df_verify.isnull().sum().sum()}")
print(f"   Duplicate rows: {df_verify.duplicated().sum()}")

print("\n" + "=" * 80)
print("RAW DATASET GENERATION COMPLETE!")
print("=" * 80)
print(f"\n✓ Dataset ready: data/customers_data.csv")
print(f"✓ Status: Contains real-world data quality issues")
print(f"✓ Next Step: EDA & Cleaning in Day 1 Part 2")
print("\nNote: This dataset has NULL values, duplicates, outliers, and")
print("inconsistencies - just like real e-commerce data!")