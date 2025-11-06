import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

#create directories
if not os.path.exists('cleaned_data'):
    os.makedirs('cleaned_data')
if not os.path.exists('output'):
    os.makedirs('output')

#load dataset
try:
    df = pd.read_csv('cleaned_data/customers_data_cleaned.csv')
    print(f"Dataset Loaded Directly from Folder")
except FileNotFoundError:
    df = pd.read_csv('../cleaned_data/customers_data_cleaned.csv')
    print(f"Dataset Loaded from Parent Directory")

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

#preprocessing the Data
print("Converting Date Columns")
df['registration_date'] = pd.to_datetime(df['registration_date'])
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

print("Date Columns converted")

#RFM Analysis RECENCY FREQUENCY MONETARY
print("RFM Analysis")

# Reference date (assume today is max date in data)
reference_date = df['last_purchase_date'].max() + pd.Timedelta(days=1)
print(f"Reference date: {reference_date}")

#calculate RFM metrics

#RECENCY: Days since last purchase
df['recency'] = (reference_date - df['last_purchase_date']).dt.days

# FREQUENCY: Number of purchases
df['frequency'] = df['purchase_frequency']

#MONETARY: Total Spending
df['monetary'] = df['total_spending']

print(f"Recency - Range: {df['recency'].min()} to {df['recency'].max()} days")
print(f"Frequency - Range: {df['frequency'].min()} to {df['frequency'].max()}")
print(f"Monetary - Range: ${df['monetary'].min():.2f} to ${df['monetary'].max():.2f}")

#RFM Scoring
print("Creating RFM SCores (1-5 scale)...")

#Receny Scoring: Lower receny is better (lower = more recent)
df['recency_score'] = pd.qcut(df['recency'],q=5,labels=[5,4,3,2,1],duplicates='drop')
df['recency_score'] = df['recency_score'].astype(int)

#frequency scoring: Higher frequency is better
df['frequency_score'] = pd.qcut(df['frequency'].rank(method='first'),q=5,labels=[1,2,3,4,5],duplicates='drop')
df['frequency_score'] = df['frequency_score'].astype(int)

#Monetary SCoring: Higher monetry is better
df['monetary_score'] = pd.qcut(df['monetary'].rank(method='first'),q=5,labels=[1,2,3,4,5],duplicates='drop')
df['monetary_score'] = df['monetary_score'].astype(int)

#combined RFM Score
df['rfm_score'] = df['recency_score'].astype(str) + df['frequency_score'].astype(str) + df['monetary_score'].astype(str)
df['rfm_combined'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']

print("RFM Scores created(1-5 scale)")
print(f"Combined score range:{df['rfm_combined'].min()} to {df['rfm_combined'].max()}")

#customer  segmentation

print("Customer Segmentation")

def segment_customer(row):
    if row['recency_score'] >=4 and row['frequency_score']>=4 and row['monetary_score']>=4:
        return 'Champions'
    elif row['recency_score'] >=4 and row['frequency_score'] >= 3 and row['monetary_score']>=3:
        return 'Loyal Customers'
    elif row['recency_score'] >=3 and row['monetary_score'] >=4:
        return 'Potential Loyalists'
    elif row['recency_score'] <= 2 and row['frequency_score']>=3:
        return 'At Risk'
    elif row['recency_score'] <= 2:
        return 'Lost'
    else:
        return 'Need Attention'

df['segment'] = df.apply(segment_customer,axis=1)

print("\n Customer Segments")
print(df['segment'].value_counts())
print(f"\nSegment percentages:")
print(df['segment'].value_counts(normalize=True)*100)

# Additional Freatures

print("Additional Features")

#Customer lifetime in days
df['customer_lifetime'] = (df['last_purchase_date'] - df['registration_date']).dt.days

#Average days between purchases
df['avg_days_between_purchases'] = df['customer_lifetime']/(df['frequency'] +1)

#Spending velocity (spending per day)
df['spending_velocity'] = df['monetary']/(df['customer_lifetime']+1)

#average spending per purchase
df['avg_spending_per_purchase'] = df['monetary']/(df['frequency']+1)

#Purchase receny in month
df['recency_months'] = df['recency']/30

#Purchase frequency per month
df['frequency_per_month'] = (df['frequency'])/(df['customer_lifetime'] + 1)

#One-hot encode product categories
print("\n Encoding categorical features")
category_dummies = pd.get_dummies(df['product_category'],prefix='category')
df = pd.concat([df,category_dummies],axis=1)
print(f"Product categories encoded: {list(category_dummies.columns)}")

#One hot encoding gender
gender_dummies = pd.get_dummies(df['gender'],prefix='gender')
df = pd.concat([df,gender_dummies],axis=1)
print(f"Gender encoded: {list(gender_dummies.columns)}")

print("\n Additonal features created")
print(f"Total features now: {len(df.columns)}")

#Feature Scaling (Normalization)
print("Feature Scaling")

from sklearn.preprocessing import MinMaxScaler

#select numeric columns to scale
numeric_features = ['age', 'recency', 'frequency', 'monetary', 'num_orders',
                   'customer_lifetime', 'avg_days_between_purchases', 
                   'spending_velocity', 'avg_spending_per_purchase',
                   'recency_months', 'frequency_per_month', 'avg_order_value']

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

print(f"Scaled {len(numeric_features)} numeric features (0-1 range)")

#Feature Summary
print("Feature Engineering Summary")
print("RFM Features")

print(f"\n Features Statistics(Scaled Data):")
print(df_scaled[numeric_features].describe())

#Visualizations + RFM Analysis

print("Creating Visualizations")

fig, axes = plt.subplots(2,3,figsize=(18,10))
fig.patch.set_facecolor('white')

# Recency Distribution
axes[0,0].hist(df['recency'],bins=30,color='skyblue',edgecolor='black',alpha=0.7)
axes[0,0].set_xlabel('Recency (days)',fontsize=10)
axes[0,0].set_ylabel('Count',fontsize=10)
axes[0,0].set_title('Distribution of Recency',fontsize=12,fontweight='bold')
axes[0,0].grid(axis='y',alpha=0.3)

#Frequency Distribution
axes[0,1].hist(df['frequency'],bins=30,color='lightcoral',edgecolor='black',alpha=0.7)
axes[0,1].set_xlabel('Frequency',fontsize=10)
axes[0,1].set_ylabel('Count',fontsize=10)
axes[0,1].set_title('Distribution of Frequency',fontsize=12,fontweight='bold')
axes[0,1].grid(axis='y',alpha=0.3)

#Monetary distribution
axes[0,2].hist(df['monetary'],bins=30,color='lightgreen',edgecolor='black',alpha=0.7)
axes[0,2].set_xlabel('Monetary ($)',fontsize=10)
axes[0,2].set_ylabel('Count',fontsize=10)
axes[0,2].set_title('Distribution of Monetary Value',fontsize=12,fontweight='bold')
axes[0,2].grid(axis='y',alpha=0.3)

#RFM Combined Score
axes[1,0].hist(df['rfm_combined'],bins=30,color='skyblue',edgecolor='black',alpha=0.7)
axes[1,0].set_xlabel('RFM Combined Score',fontsize=10)
axes[1,0].set_ylabel('Count',fontsize=10)
axes[1,0].set_title('Distribution of RFM Combined Score',fontsize=12,fontweight='bold')
axes[1,0].grid(axis='y',alpha=0.3)

# Recency vs Monetary
axes[1,1].scatter(df['recency'],df['monetary'],alpha=0.5,s=.05,color='teal')
axes[1,1].set_xlabel('Recency (days)',fontsize=10)
axes[1,1].set_ylabel('Monetary ($)',fontsize=10)
axes[1,1].set_title('Recency vs Monetary',fontsize=12,fontweight='bold')
axes[1,1].grid(alpha=0.3)

#Frequency vs Monetary
axes[1,2].scatter(df['frequency'],df['monetary'],alpha=0.5,s=.05,color='orange')
axes[1,2].set_xlabel('Frequency',fontsize=10)
axes[1,2].set_ylabel('Monetary ($)',fontsize=10)
axes[1,2].set_title('Frequency vs Monetary',fontsize=12,fontweight='bold')
axes[1,2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../output/rfm_analysis.png',dpi=300,bbox_inches='tight',facecolor='white')
print('RFM analysis visualization saved')
plt.show()

# Visualizations - customer segments

fig,axes = plt.subplots(2,2,figsize=(16,12))
fig.patch.set_facecolor('white')

#segment distribution
segments_counts = df['segment'].value_counts()
colors=plt.cm.Set3(range(len(segments_counts)))
axes[0,0].bar(segments_counts.index,segments_counts.values,color=colors,edgecolor='black')
axes[0,0].set_xlabel('Segment',fontsize=10)
axes[0,0].set_ylabel('Count',fontsize=10)
axes[0,0].set_title('Customer Segment Distribution',fontsize=12,fontweight='bold')
axes[0,0].tick_params(axis='x',rotation=45)
axes[0,0].grid(axis='y',alpha=0.3)

# Segment percentage
axes[0,1].pie(segments_counts.values,labels=segments_counts.index,autopct='%1.1f%%',
        colors=colors, startangle=90)
axes[0,1].set_title('Customer Segment Percentages',fontsize=12,fontweight='bold')

# Average monetary by segment
segment_monetary = df.groupby('segment')['monetary'].mean().sort_values(ascending=False)
axes[1,0].barh(segment_monetary.index,segment_monetary.values,color='lightblue',edgecolor='black')
axes[1,0].set_xlabel('Average Monetary ($)',fontsize=10)
axes[1,0].set_title('Average Spending by Segment',fontsize=12,fontweight='bold')
axes[1,0].grid(axis='x',alpha=0.3)

#Average frequency by segment
segment_frequency = df.groupby('segment')['frequency'].mean().sort_values(ascending=False)
axes[1,1].barh(segment_frequency.index,segment_frequency.values,color='lightcoral',edgecolor='black')
axes[1,1].set_xlabel('Average Frequency',fontsize=10)
axes[1,1].set_title('Average Purchase Frequency by Segment',fontsize=12,fontweight='bold')
axes[1,1].grid(axis='x',alpha=0.3)

plt.tight_layout()
plt.savefig('../output/customer_segment.png',dpi=300,bbox_inches='tight',facecolor='white')
print('Customer segments visualization saved')
plt.show()

#saving feature Engineered data

print("Saving Feature-Engineered Data")

#Save full dataset (with all features)
rfm_csv_path = '../cleaned_data/customer_data_rfm.csv'
df.to_csv(rfm_csv_path,index=False)
print(f"Full feaure-engineered data saved:{rfm_csv_path}")

#save scaled data (for modeling)
scaled_csv_path = '../cleaned_data/customer_data_scaled.csv'
df_scaled.to_csv(scaled_csv_path,index=False)
print(f"Scaled data saved:{scaled_csv_path}")

#Summary and insight
print('Summary and Insights')


print(f"""
✅ RFM Analysis:
   - Recency: {df['recency'].mean():.1f} days (avg)
   - Frequency: {df['frequency'].mean():.1f} purchases (avg)
   - Monetary: ${df['monetary'].mean():.2f} (avg)
   - Combined RFM Score: {df['rfm_combined'].mean():.2f} (avg)

👥 Customer Segments:
   - Champions: {len(df[df['segment'] == 'Champions'])} ({len(df[df['segment'] == 'Champions'])/len(df)*100:.1f}%)
   - Loyal: {len(df[df['segment'] == 'Loyal Customers'])} ({len(df[df['segment'] == 'Loyal Customers'])/len(df)*100:.1f}%)
   - Potential: {len(df[df['segment'] == 'Potential Loyalists'])} ({len(df[df['segment'] == 'Potential Loyalists'])/len(df)*100:.1f}%)
   - At Risk: {len(df[df['segment'] == 'At Risk'])} ({len(df[df['segment'] == 'At Risk'])/len(df)*100:.1f}%)
   - Lost: {len(df[df['segment'] == 'Lost'])} ({len(df[df['segment'] == 'Lost'])/len(df)*100:.1f}%)
""")
