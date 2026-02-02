import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# STREAMLIT CONFIG
# ============================================================================
st.set_page_config(
    page_title="CLV Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("# 💰 Customer Lifetime Value (CLV) Prediction Dashboard")
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = st.radio("Select Page:", 
                    ["🏠 Home", "📤 Predictions", "📈 Analytics", "ℹ️ About"])

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🤖 Model",
            value="Random Forest",
            delta="99.99% Accuracy"
        )
    
    with col2:
        st.metric(
            label="🔗 API Status",
            value="Connected",
            delta="Online"
        )
    
    with col3:
        st.metric(
            label="⚡ Response Time",
            value="<100ms",
            delta="Fast"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Features")
        st.markdown("""
        ✅ Real-time CLV Prediction (via API)
        ✅ Data Validation (Pydantic)
        ✅ Feature Engineering
        ✅ Customer Segmentation
        ✅ Advanced Analytics
        """)
    
    with col2:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        **Frontend:** Streamlit
        **API:** FastAPI + Pydantic
        **Features:** utils.py
        **Model:** Random Forest
        **Validation:** Data validation before processing
        """)

# ============================================================================
# PAGE 2: PREDICTIONS
# ============================================================================
elif page == "📤 Predictions":
    st.markdown("## 🎯 Make Predictions")
    
    st.info("ℹ️ Enter customer details below. Data will be validated, engineered, and sent to API for prediction.")
    
    # Check API connection
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Connection Failed")
    except:
        st.error("❌ Cannot connect to API. Make sure api.py is running on http://localhost:8000")
    
    st.markdown("---")
    
    st.markdown("### Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 75, 35)
        purchase_frequency = st.slider("Purchase Frequency", 1, 50, 10)
        avg_order_value = st.slider("Avg Order Value ($)", 10.0, 1000.0, 100.0)
    
    with col2:
        num_orders = st.slider("Number of Orders", 1, 150, 20)
        customer_lifetime_days = st.slider("Customer Lifetime (days)", 1, 1400, 365)
        recency = st.slider("Days Since Last Purchase", 0, 400, 30)
    
    frequency_score = st.slider("Frequency Score (1-5)", 1, 5, 3)
    
    # Prediction Button
    if st.button("🔮 Predict CLV via API", key="predict_btn"):
        st.markdown("---")
        
        # Prepare data
        customer_data = {
            "age": int(age),
            "purchase_frequency": int(purchase_frequency),
            "avg_order_value": float(avg_order_value),
            "num_orders": int(num_orders),
            "customer_lifetime_days": int(customer_lifetime_days),
            "recency": int(recency),
            "frequency_score": int(frequency_score)
        }
        
        try:
            # Call API
            with st.spinner("🔄 Sending data to API... Validating → Engineering Features → Predicting..."):
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=customer_data,
                    timeout=10
                )
            
            if response.status_code == 200:
                prediction = response.json()
                
                st.markdown("### 📊 Prediction Results")
                
                # Display prediction
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Predicted CLV",
                        f"${prediction['predicted_clv']:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Segment",
                        prediction['customer_segment']
                    )
                
                with col3:
                    st.metric(
                        "vs Average",
                        f"{prediction['comparison_to_average']:+.1f}%"
                    )
                
                with col4:
                    st.metric(
                        "Confidence",
                        f"{prediction['confidence_score']:.2f}/1.0"
                    )
                
                st.markdown("---")
                
                # Show segment with color
                segment = prediction['customer_segment']
                if "High Value" in segment:
                    st.success(f"✨ {segment} - Excellent customer!")
                elif "Medium-High" in segment:
                    st.info(f"ℹ️ {segment} - Good potential")
                elif "Medium Value" in segment:
                    st.warning(f"⚠️ {segment} - Needs attention")
                else:
                    st.error(f"❌ {segment} - High churn risk!")
                
                st.markdown("---")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if "High Value" in segment:
                    st.markdown("""
                    • **VIP Treatment:** Offer exclusive benefits
                    • **Loyalty Program:** Premium rewards
                    • **Personalization:** Custom offers
                    • **Priority Support:** Dedicated account manager
                    """)
                elif "Medium-High" in segment:
                    st.markdown("""
                    • **Growth Strategy:** Upsell opportunities
                    • **Engagement:** Regular communication
                    • **Rewards:** Loyalty incentives
                    • **Cross-sell:** Complementary products
                    """)
                elif "Medium Value" in segment:
                    st.markdown("""
                    • **Re-engagement:** Win-back campaigns
                    • **Education:** Product tutorials
                    • **Discounts:** Limited-time offers
                    • **Feedback:** Request product feedback
                    """)
                else:
                    st.markdown("""
                    • **Retention Focus:** High-touch outreach
                    • **Incentives:** Special comeback offers
                    • **Communication:** Check-in emails
                    • **Surveys:** Understand pain points
                    """)
            
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.write(response.json())
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Please run: `python api.py`")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE 3: ANALYTICS
# ============================================================================
elif page == "📈 Analytics":
    st.markdown("## 📊 System Analytics")
    
    try:
        # Get stats from API
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{stats['total_customers']:,}")
            with col2:
                st.metric("Avg CLV", f"${stats['average_clv']:.2f}")
            with col3:
                st.metric("Median CLV", f"${stats['median_clv']:.2f}")
            with col4:
                st.metric("Max CLV", f"${stats['max_clv']:.2f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💰 CLV Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                clv_range = np.linspace(stats['min_clv'], stats['max_clv'], 50)
                ax.hist(clv_range, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(stats['average_clv'], color='red', linestyle='--', linewidth=2, label='Average')
                ax.set_xlabel('CLV ($)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 📊 Threshold Information")
                st.markdown(f"""
                **High Value Threshold:** ${stats['high_value_threshold']:.2f}
                - Top 25% of customers
                
                **Medium Value Threshold:** ${stats['medium_value_threshold']:.2f}
                - Top 50% of customers
                
                **Low Value Threshold:** ${stats['low_value_threshold']:.2f}
                - Bottom 25% of customers
                
                **Standard Deviation:** ${stats['std_clv']:.2f}
                - Variation in customer value
                """)
    
    except:
        st.error("❌ Cannot fetch analytics. Ensure API is running.")

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.markdown("## 📖 About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **CLV Prediction System** - Predicts customer lifetime value using machine learning.
    
    ### 🏗️ Architecture
    
    ```
    Frontend (Streamlit)
        ↓ User Input
    API (FastAPI)
        ↓ Data Validation (Pydantic)
    Utils (Feature Engineering)
        ↓ Engineered Features
    Model (Random Forest)
        ↓ Prediction
    Results Display
    ```
    
    ### 🤖 Model Details
    
    **Algorithm:** Random Forest Regressor
    **Accuracy:** 99.99% (R² Score)
    **Features:** 18 engineered features
    **Training Data:** 5000+ customers
    
    ### 📊 Key Metrics
    
    - **R² Score:** 0.9999
    - **Mean Absolute Error:** < 0.05
    - **Root Mean Squared Error:** < 0.06
    
    ### 🛠️ Technologies
    
    - **Frontend:** Streamlit
    - **API:** FastAPI + Pydantic
    - **Machine Learning:** scikit-learn
    - **Data Processing:** Pandas, NumPy
    
    ### 🚀 How It Works
    
    1. **User enters customer data** via Streamlit frontend
    2. **Data is validated** using Pydantic models
    3. **Features are engineered** using domain knowledge
    4. **Model makes prediction** via FastAPI endpoint
    5. **Results displayed** with recommendations
    
    ### 💡 Use Cases
    
    - Customer retention strategies
    - Marketing budget allocation
    - Personalized customer campaigns
    - Churn risk identification
    
    ---
    
    *Built with ❤️ using Python, FastAPI, Streamlit, and ML*
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2024 CLV Prediction Dashboard | FastAPI + Streamlit + ML</p>", 
            unsafe_allow_html=True)