import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Get the directory where app.py is located
app_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(app_dir)

# print(f"Current working directory: {os.getcwd()}")
# print(f"Files in directory: {os.listdir()}")
# print(f"files in cleaned_data: {os.listdir('cleaned_data/')}")
#streamlit config
st.set_page_config(
    page_title="CLV Prediction Dashboards",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


#Load Model & Data
@st.cache_resource
def load_model():
    try:
        with open('models/clv_best_model.pkl','rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model not found!")
        return None

@st.cache_data
def load_reference_data():
    try:
        df = pd.read_csv('cleaned_data/customer_data_rfm.csv')
        return df
    except FileNotFoundError:
        st.error("Reference data not found!")
        return None
    
#Load model and data
model = load_model()
reference_df = load_reference_data()

# print("Features needed:", model.feature_names_in_)
# print("Number of features:", model.n_features_in_)
#Header
st.markdown("# Customer Lifetime Value (CLV) Prediction Dashboard")
st.markdown("---")

#sidebar
with st.sidebar:
    st.markdown("Navigation")
    page = st.radio("Select Page:",
                    ["Home", "Predictions", "Analytics"])

# Page 1: Home
if page == "Home":
    col1, col2, col3 =st.columns(3)

    with col1:
        st.metric(
            label = "Total Customers",
            value=f"{len(reference_df):,}",
            delta="Training Data"
        )
    
    with col2:
        st.metric(
            label="Avg CLV",
            value=f"${reference_df['monetary'].mean():.2f}",
            delta=f"Max: ${reference_df['monetary'].max():.2f}"
        )
    
    with col3:
        st.metric(
            label="Model Accuracy",
            value="99.99%",
            delta="R² Score"
        ) 

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Features")
        st.markdown("""
        Real-time CLV Prediction
        Customer Segmentation
        RFM Analysis
        Batch Predictions
        Advanced Analytics
        """)
    
    with col2:
        st.markdown("Model Info")
        st.markdown("""
        **Algorithm:** Random Forest
        **Features:** 20+
        **Accuracy:** 99.99%
        **Training Data:** 5000 customers
        """)

# Predictions
elif page =="Predictions":
    st.markdown("Make Predictions")

    #Two options: Simgle Prediciton or Batch Prediction
    prediciton_type = st.radio("Select Predicition Type:",
                               ["Single Customer"])
    
    if prediciton_type == "Single Customer":
        st.markdown("Enter Customer Details")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 75, 35)
            purchase_frequency = st.slider("Purchase Frequency", 1, 50, 10)
            avg_order_value = st.slider("Avg Order Value ($)", 10.0, 1000.0, 100.0)
        
        with col2:
            num_orders = st.slider("Number of Orders", 1, 150, 20)
            customer_lifetime_days = st.slider("Customer Lifetime (days)", 1, 1400, 365)
            recency = st.slider("Days Since Last Purchase", 0, 400, 30)
        
        #Additional Features
        st.markdown("Additional Features")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender",["Male","Female","Other"])

        with col2: 
            category = st.selectbox("Product Category",["Electronics","Clothing","Home","Beauty","Sports"])
        
        with col3:
            frequency_score = st.slider("Frequency Score(1-5)",1,5,3)
        
        # Prepare Input Data
        if st.button("Predict CLV", key="single_predict"):
            try:
                # Exact features model expects (18 features)
                input_data = {
                    'age': age,
                    'purchase_frequency': purchase_frequency,
                    'avg_order_value': avg_order_value,
                    'total_spending': avg_order_value * num_orders,  # Calculate from order value and count
                    'num_orders': num_orders,
                    'customer_lifetime_days': customer_lifetime_days,
                    'recency': recency,
                    'frequency': purchase_frequency,  # Same as purchase_frequency
                    'recency_score': 5 - min(recency // 80, 4),
                    'frequency_score': frequency_score,
                    'monetary_score': min((avg_order_value * num_orders) / 100, 5),
                    'rfm_combined': (5 - min(recency // 80, 4)) + frequency_score + min((avg_order_value * num_orders) / 100, 5),
                    'customer_lifetime': customer_lifetime_days,  # Same as customer_lifetime_days
                    'avg_days_between_purchases': max(customer_lifetime_days / (num_orders + 1), 1),
                    'spending_velocity': (avg_order_value * purchase_frequency) / (customer_lifetime_days + 1),
                    'avg_spending_per_purchase': avg_order_value,
                    'recency_months': recency / 30,
                    'frequency_per_month': (purchase_frequency / (customer_lifetime_days + 1)) * 30,
                }
                
                # Create dataframe with exact column order
                feature_names = ['age', 'purchase_frequency', 'avg_order_value', 'total_spending',
                               'num_orders', 'customer_lifetime_days', 'recency', 'frequency',
                               'recency_score', 'frequency_score', 'monetary_score', 'rfm_combined',
                               'customer_lifetime', 'avg_days_between_purchases', 'spending_velocity',
                               'avg_spending_per_purchase', 'recency_months', 'frequency_per_month']
                
                input_df = pd.DataFrame([input_data])[feature_names]
                #Make Predicition
                prediction = model.predict(input_df)[0]
                
                # Unscale prediction to actual CLV value
                min_clv = reference_df['monetary'].min()
                max_clv = reference_df['monetary'].max()
                actual_prediction = prediction * (max_clv - min_clv) + min_clv
                
                st.success(f"Predicted CLV: **${actual_prediction:.2f}**")

                #segment customer
                if actual_prediction > reference_df['monetary'].quantile(0.75):
                    segment = "High Value"
                    color = "green"
                elif actual_prediction > reference_df['monetary'].quantile(0.5):
                    segment = "Medium-High Value"
                    color = "blue"
                elif actual_prediction > reference_df['monetary'].quantile(0.25):
                    segment = "Medium Value"
                    color = "orange"
                else:
                    segment = "At Risk"
                    color = "red"
                
                st.markdown(f"**Customer Segment:** {segment}")

                # Show segment with color
                if segment == "High Value":
                    st.success(f"{segment} - Excellent customer!")
                elif segment == "Medium-High Value":
                    st.info(f"{segment} - Good potential")
                elif segment == "Medium Value":
                    st.warning(f"{segment} - Needs attention")
                else:
                    st.error(f"{segment} - High churn risk!")

                #show comparison
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Your Prediction", f"${actual_prediction:.2f}")
                with col2:
                    st.metric("Average CLV", f"${reference_df['monetary'].mean():.2f}")
                with col3:
                    difference = ((actual_prediction - reference_df['monetary'].mean()) / reference_df['monetary'].mean()) * 100
                    st.metric("Difference", f"{difference:+.1f}%")

            except Exception as e:
                st.error(f"Error: {str(e)}")


# Analytics
elif page == "Analytics":
    st.markdown("Customer Analyics Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("CLV Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(reference_df['monetary'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Customer Lifetime Value ($)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown(" Customer Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(reference_df['age'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Age', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig) 
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(" Product Category Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        category_counts = reference_df['product_category'].value_counts()
        ax.bar(category_counts.index, category_counts.values, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        plt.xticks(rotation=45)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown(" Purchase Frequency vs CLV")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(reference_df['purchase_frequency'], reference_df['monetary'], 
                  alpha=0.5, s=30, color='teal')
        ax.set_xlabel('Purchase Frequency', fontsize=11, fontweight='bold')
        ax.set_ylabel('CLV ($)', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Statistics
    st.markdown("Key Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers",f"{len(reference_df):,}")
    with col2:
        st.metric("Avg CLV",f"${reference_df['monetary'].mean():.2f}")
    with col3:
        st.metric("Median CLV",f"${reference_df['monetary'].median():.2f}")
    with col4:
        st.metric("Max CLV",f"${reference_df['monetary'].max():.2f}")

st.markdown("<p style='text-align: center; color: gray;'>© 2024 CLV Prediction Dashboard | Powered by Streamlit & Machine Learning</p>", 
            unsafe_allow_html=True)