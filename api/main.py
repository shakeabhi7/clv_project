from fastapi import FastAPI, HTTPException, Request,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import os
import pickle
from contextlib import asynccontextmanager
from datetime import datetime
from backend.database import PredictionDatabase

from backend.utils import (
    engineer_features,
    unscale_prediction,
    segment_customer,
    get_confidence_score,
    calculate_comparison
)



# ── CONFIG FROM ENVIRONMENT (Docker-friendly)
MODEL_PATH = os.getenv("MODEL_PATH", "models/clv_best_model.pkl")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "cleaned_data/customer_data_rfm.csv")


# FASTAPI LIFESPAN (STARTUP / SHUTDOWN)
@asynccontextmanager
async def lifespan(app: FastAPI):
    #StartUp
    print("Starting CLV API...")

    try:
        with open(MODEL_PATH,'rb') as f:
            app.state.model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")
    
    try:
        app.state.reference_df = pd.read_csv(REFERENCE_DATA_PATH)
        print(f"Reference data loaded: {len(app.state.reference_df)} rows")
    except Exception as e:
        raise RuntimeError(f"Reference data loading failed: {e}")
    
    try:
        app.state.db = PredictionDatabase()
        print("Database initialized")
    except Exception as e:
        app.state.db = None
        print(f"[WARNING] Database initialization failed: {e} — predictions will not be logged")

    yield

    #Shutdown
    if app.state.db:
        app.state.db.close()
    print("CLV API shutdown complete")


# FASTAPI APP SETUP

app = FastAPI(
    title="CLV Prediction API",
    description="Customer Lifetime Value Prediction API",
    version="1.0.0",
    lifespan=lifespan

)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Dependencies
def get_db(request: Request) -> PredictionDatabase:
    db = request.app.state.db
    if db is None:
        raise HTTPException(500, "Database not initialized")
    return db


def get_model(request: Request):
    model = request.app.state.model
    if model is None:                                     
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model
 
 
def get_df(request: Request) -> pd.DataFrame:
    df = request.app.state.reference_df
    if df is None:                                        
        raise HTTPException(status_code=503, detail="Reference data not loaded")
    return 


# PYDANTIC MODELS (DATA VALIDATION)
class CustomerInput(BaseModel):
    """Single customer CLV prediction input with validation"""
    age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    purchase_frequency: int = Field(..., ge=1, le=50, description="Purchase frequency (1-50)")
    avg_order_value: float = Field(..., ge=10.0, le=1000.0, description="Average order value ($)")
    num_orders: int = Field(..., ge=1, le=150, description="Number of orders (1-150)")
    customer_lifetime_days: int = Field(..., ge=1, le=1400, description="Customer lifetime in days")
    recency: int = Field(..., ge=0, le=400, description="Days since last purchase")
    frequency_score: int = Field(..., ge=1, le=5, description="Frequency score (1-5)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35, 
                "purchase_frequency": 20, 
                "avg_order_value": 150.0,
                "num_orders": 25, 
                "customer_lifetime_days": 365,
                "recency": 30, 
                "frequency_score": 4
            }
        }
    }


class CLVPredictionResponse(BaseModel):
    """CLV prediction response"""
    predicted_clv: float
    customer_segment: str
    comparison_to_average: float
    confidence_score: float
    
    model_config = {
        "json_schema_extra" : {
            "example": {
                "predicted_clv": 8500.50,
                "customer_segment": "High Value",
                "comparison_to_average": 11.3,
                "confidence_score": 0.95
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    data_loaded: bool
    total_training_records: int
    timestamp: str



# API ENDPOINTS
@app.get("/", tags=["Info"])
def read_root():
    """Root endpoint with API info"""
    return {
        "message": "CLV Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check(model=Depends(get_model),df=Depends(get_df)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded":  df is not None,
        "total_training_records": len(df),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=CLVPredictionResponse, tags=["Predictions"])
def predict_clv(customer: CustomerInput, model=Depends(get_model),df=Depends(get_df),db=Depends(get_db)):
    """Predict Customer Lifetime Value"""

    try:
        # Step 1: Data is validated by Pydantic automatically
        
        # Step 2: Feature Engineering (utils.py)
        customer_dict = customer.model_dump()
        engineered_features = engineer_features(customer_dict)
        
        # Step 3: Model Prediction
        scaled_pred = model.predict(engineered_features)[0]
        
        # Step 4: Post-processing
        actual_clv = unscale_prediction(scaled_pred, df)
        segment = segment_customer(actual_clv, df)
        comparison = calculate_comparison(actual_clv, df)
        confidence = get_confidence_score(actual_clv, df)

        #step 5
        if db is not None:
            db.save_prediction(
                input_data=customer_dict,
                engineered_features=engineered_features.to_dict(orient='records')[0],
                scaled_prediction=float(scaled_pred),
                predicted_clv=float(actual_clv),
                customer_segment=segment,
                comparison_to_average=float(comparison),
                confidence_score=float(confidence)
            )
        
        return {
            "predicted_clv": round(actual_clv, 2),
            "customer_segment": segment,
            "comparison_to_average": round(comparison, 2),
            "confidence_score": round(confidence, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/stats", tags=["Info"])
def get_stats(df=Depends(get_df)):
    """Get training data statistics"""
    
    return {
        "total_customers": len(df),
        "average_clv": round(df['monetary'].mean(), 2),
        "median_clv": round(df['monetary'].median(), 2),
        "max_clv": round(df['monetary'].max(), 2),
        "min_clv": round(df['monetary'].min(), 2),
        "std_clv": round(df['monetary'].std(), 2),
        "high_value_threshold": round(df['monetary'].quantile(0.75), 2),
        "medium_value_threshold": round(df['monetary'].quantile(0.5), 2),
        "low_value_threshold": round(df['monetary'].quantile(0.25), 2)
    }



@app.get("/database/stats",tags=["Database"])
def get_database_stats(db=Depends(get_db)):
    """Get prediction stats from database"""
    return db.get_statistics()



@app.get("/database/predictions",tags=["Database"])
def get_all_predictions(db=Depends(get_db),limit:int=100):
    """Get all predictions from database"""
    predictions = db.get_all_predictions(limit)
    return {
        "total_records":len(predictions),
        "predictions":predictions
    }


@app.get("/database/segment/{segment}", tags=["Database"])
def get_predictions_by_segment(segment: str,db=Depends(get_db) ):
    """Get predictions by customer segment"""
    predictions = db.get_predictions_by_segment(segment)
    return {
        "segment": segment,
        "total_records": len(predictions),
        "predictions": predictions
    }


@app.post("/database/export", tags=["Database"])
def export_predictions(filename: str = "clv_predictions.csv",db=Depends(get_db)):
    """Export all predictions to CSV"""
    success = db.export_to_csv(filename)
    return {
        "success": success,
        "filename": filename,
        "message": "Predictions exported successfully" if success else "Export failed"
    }


@app.delete("/database/prediction/{prediction_id}", tags=["Database"],)
def delete_prediction_by_id(prediction_id: str,db=Depends(get_db)):
    """
    Delete a specific prediction by MongoDB ObjectId
    
    Args:
        prediction_id: MongoDB ObjectId (24-character hex string)
        
    Returns:
        Success/failure status
    """
    success = db.delete_prediction_by_id(prediction_id)
    return {
        "success": success,
        "prediction_id": prediction_id,
        "message": f"Prediction {prediction_id} deleted successfully" if success else f"Failed to delete prediction {prediction_id}"
    }

@app.delete("/database/clear", tags=["Database"])
def clear_database(db=Depends(get_db)):
    """Clear all predictions from database (use with caution!)"""
    success = db.clear_all_predictions()
    return {
        "success": success,
        "message": "Database cleared" if success else "Clear failed"
    }



@app.get("/database/info", tags=["Database"])
def get_database_info(db=Depends(get_db)):
    """Get database file information"""
    return db.get_database_info()
