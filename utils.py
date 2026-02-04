import pandas as pd
import numpy as np

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def engineer_features(customer_data: dict) -> pd.DataFrame:
    """
    Engineer features from raw customer input
    
    Args:
        customer_data: Dictionary with raw customer information
        
    Returns:
        DataFrame with engineered features matching model training data
    """
    
    # Extract input values
    age = customer_data.get('age')
    purchase_frequency = customer_data.get('purchase_frequency')
    avg_order_value = customer_data.get('avg_order_value')
    num_orders = customer_data.get('num_orders')
    customer_lifetime_days = customer_data.get('customer_lifetime_days')
    recency = customer_data.get('recency')
    frequency_score = customer_data.get('frequency_score')
    
    # Calculate derived features
    total_spending = avg_order_value * num_orders
    
    recency_score = 5 - min(recency // 80, 4)
    monetary_score = min((avg_order_value * num_orders) / 100, 5)
    
    rfm_combined = recency_score + frequency_score + monetary_score
    
    customer_lifetime = customer_lifetime_days
    
    avg_days_between_purchases = max(customer_lifetime_days / (num_orders + 1), 1)
    
    spending_velocity = (avg_order_value * purchase_frequency) / (customer_lifetime_days + 1)
    
    avg_spending_per_purchase = avg_order_value
    
    recency_months = recency / 30
    
    frequency_per_month = (purchase_frequency / (customer_lifetime_days + 1)) * 30
    
    frequency = purchase_frequency
    
    # Create feature dictionary
    features_dict = {
        'age': age,
        'purchase_frequency': purchase_frequency,
        'avg_order_value': avg_order_value,
        'total_spending': total_spending,
        'num_orders': num_orders,
        'customer_lifetime_days': customer_lifetime_days,
        'recency': recency,
        'frequency': frequency,
        'recency_score': recency_score,
        'frequency_score': frequency_score,
        'monetary_score': monetary_score,
        'rfm_combined': rfm_combined,
        'customer_lifetime': customer_lifetime,
        'avg_days_between_purchases': avg_days_between_purchases,
        'spending_velocity': spending_velocity,
        'avg_spending_per_purchase': avg_spending_per_purchase,
        'recency_months': recency_months,
        'frequency_per_month': frequency_per_month,
    }
    
    # Feature order (must match model training order)
    feature_order = [
        'age', 'purchase_frequency', 'avg_order_value', 'total_spending',
        'num_orders', 'customer_lifetime_days', 'recency', 'frequency',
        'recency_score', 'frequency_score', 'monetary_score', 'rfm_combined',
        'customer_lifetime', 'avg_days_between_purchases', 'spending_velocity',
        'avg_spending_per_purchase', 'recency_months', 'frequency_per_month'
    ]
    
    # Create DataFrame with correct order
    features_df = pd.DataFrame([features_dict])[feature_order]
    
    return features_df


def unscale_prediction(scaled_pred: float, reference_df: pd.DataFrame) -> float:
    """
    Convert scaled prediction to actual CLV value
    
    Args:
        scaled_pred: Prediction from model (0-1 range)
        reference_df: Reference data for scaling information
        
    Returns:
        Actual CLV value
    """
    min_clv = reference_df['monetary'].min()
    max_clv = reference_df['monetary'].max()
    return scaled_pred * (max_clv - min_clv) + min_clv


def segment_customer(clv_value: float, reference_df: pd.DataFrame) -> str:
    """
    Segment customer based on CLV value
    
    Args:
        clv_value: Predicted CLV value
        reference_df: Reference data for quantile calculation
        
    Returns:
        Customer segment string with emoji
    """
    if clv_value > reference_df['monetary'].quantile(0.75):
        return "High Value"
    elif clv_value > reference_df['monetary'].quantile(0.5):
        return "Medium-High Value"
    elif clv_value > reference_df['monetary'].quantile(0.25):
        return "Medium Value"
    else:
        return "At Risk"


def get_confidence_score(clv_value: float, reference_df: pd.DataFrame) -> float:
    """
    Calculate confidence score based on CLV value range
    
    Args:
        clv_value: Predicted CLV value
        reference_df: Reference data for range calculation
        
    Returns:
        Confidence score (0-1)
    """
    q25 = reference_df['monetary'].quantile(0.25)
    q75 = reference_df['monetary'].quantile(0.75)
    
    if q25 <= clv_value <= q75:
        return 0.95
    elif clv_value > q75 or clv_value < q25:
        return 0.85
    else:
        return 0.75


def calculate_comparison(clv_value: float, reference_df: pd.DataFrame) -> float:
    """
    Calculate % difference from average CLV
    
    Args:
        clv_value: Predicted CLV value
        reference_df: Reference data for average calculation
        
    Returns:
        Percentage difference
    """
    avg_clv = reference_df['monetary'].mean()
    return ((clv_value - avg_clv) / avg_clv) * 100