from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime, timedelta
from typing import Dict, List
import json


# MONGODB CONFIGURATION


MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "clv_predictions"
COLLECTION_NAME = "predictions"


# MONGODB CONNECTION & SETUP


def get_mongo_client():
    """Get MongoDB client connection"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        print("✓ MongoDB connected successfully")
        return client
    except ServerSelectionTimeoutError:
        print("✗ MongoDB connection failed. Make sure MongoDB is running on localhost:27017")
        return None
    except ConnectionFailure:
        print("✗ MongoDB connection error")
        return None


def init_database():
    """Initialize MongoDB database and collections"""
    try:
        client = get_mongo_client()
        if not client:
            return False
        
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Create indexes for better performance
        collection.create_index("timestamp")
        collection.create_index("customer_segment")
        collection.create_index("predicted_clv")
        
        print("✓ MongoDB database and indexes initialized")
        client.close()
        return True
    
    except Exception as e:
        print(f"✗ Database initialization error: {str(e)}")
        return False



# DATABASE OPERATIONS


class PredictionDatabase:
    """Database operations for CLV predictions using MongoDB"""
    
    def __init__(self, 
                 mongodb_uri: str = MONGODB_URI,
                 db_name: str = DATABASE_NAME,
                 collection_name: str = COLLECTION_NAME):
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            print("✓ Connected to MongoDB")
        except Exception as e:
            print(f"✗ MongoDB connection error: {str(e)}")
            self.client = None
            self.db = None
            self.collection = None
    
    def save_prediction(self, 
                       input_data: Dict,
                       engineered_features: Dict,
                       scaled_prediction: float,
                       predicted_clv: float,
                       customer_segment: str,
                       comparison_to_average: float,
                       confidence_score: float) -> bool:
        """
        Save prediction to MongoDB
        
        Args:
            input_data: Raw input from user
            engineered_features: Engineered features dict
            scaled_prediction: Prediction from model (0-1)
            predicted_clv: Unscaled actual CLV
            customer_segment: Segment classification
            comparison_to_average: % difference from average
            confidence_score: Model confidence
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if self.collection is None:
                print("✗ MongoDB not connected")
                return False
            
            # Create prediction document
            prediction_doc = {
                "timestamp": datetime.now(),
                
                # Input Features
                "age": input_data.get('age'),
                "purchase_frequency": input_data.get('purchase_frequency'),
                "avg_order_value": input_data.get('avg_order_value'),
                "num_orders": input_data.get('num_orders'),
                "customer_lifetime_days": input_data.get('customer_lifetime_days'),
                "recency": input_data.get('recency'),
                "frequency_score": input_data.get('frequency_score'),
                
                # Engineered Features
                "engineered_features": engineered_features,
                
                # Predictions
                "scaled_prediction": float(scaled_prediction),
                "predicted_clv": float(predicted_clv),
                "customer_segment": customer_segment,
                "comparison_to_average": float(comparison_to_average),
                "confidence_score": float(confidence_score),
                
                # Metadata
                "model_version": "1.0"
            }
            
            result = self.collection.insert_one(prediction_doc)
            print(f"✓ Prediction saved to MongoDB (ID: {result.inserted_id}, CLV: ${predicted_clv:.2f})")
            return True
        
        except Exception as e:
            print(f"✗ Error saving prediction: {str(e)}")
            return False
    
    def get_all_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Get all predictions from MongoDB
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if self.collection is None:
                print("✗ MongoDB not connected")
                return []
            
            predictions = list(self.collection.find().sort("timestamp", -1).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for pred in predictions:
                pred['_id'] = str(pred['_id'])
            
            return predictions
        
        except Exception as e:
            print(f"✗ Error fetching predictions: {str(e)}")
            return []
    
    def get_predictions_by_segment(self, segment: str) -> List[Dict]:
        """
        Get predictions filtered by customer segment
        
        Args:
            segment: Customer segment to filter by
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if self.collection is None:
                print("✗ MongoDB not connected")
                return []
            
            predictions = list(self.collection.find(
                {"customer_segment": segment}
            ).sort("timestamp", -1))
            
            for pred in predictions:
                pred['_id'] = str(pred['_id'])
            
            return predictions
        
        except Exception as e:
            print(f"✗ Error fetching segment predictions: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get overall statistics from MongoDB
        
        Returns:
            Dictionary with statistics
        """
        try:
            if self.collection is None:
                print("✗ MongoDB not connected")
                return {}
            
            # Aggregation pipeline for statistics
            stats_pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_predictions": {"$sum": 1},
                        "average_clv": {"$avg": "$predicted_clv"},
                        "average_confidence": {"$avg": "$confidence_score"},
                        "max_clv": {"$max": "$predicted_clv"},
                        "min_clv": {"$min": "$predicted_clv"}
                    }
                }
            ]
            
            stats = list(self.collection.aggregate(stats_pipeline))
            
            # Segment distribution
            segment_pipeline = [
                {
                    "$group": {
                        "_id": "$customer_segment",
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            segments = list(self.collection.aggregate(segment_pipeline))
            segment_dict = {seg['_id']: seg['count'] for seg in segments}
            
            if stats:
                return {
                    "total_predictions": int(stats[0].get('total_predictions', 0)),
                    "average_clv": round(stats[0].get('average_clv', 0), 2),
                    "average_confidence": round(stats[0].get('average_confidence', 0), 2),
                    "max_clv": round(stats[0].get('max_clv', 0), 2),
                    "min_clv": round(stats[0].get('min_clv', 0), 2),
                    "segment_distribution": segment_dict,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "total_predictions": 0,
                    "average_clv": 0,
                    "average_confidence": 0,
                    "segment_distribution": {},
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            print(f"✗ Error getting statistics: {str(e)}")
            return {}
    
    def export_to_csv(self, filename: str = "clv_predictions.csv") -> bool:
        """
        Export predictions to CSV file
        
        Args:
            filename: Output CSV filename
            
        Returns:
            True if exported successfully
        """
        try:
            import pandas as pd
            
            if self.collection is None:
                print("✗ MongoDB not connected")
                return False
            
            predictions = list(self.collection.find().sort("timestamp", -1))
            
            # Convert to DataFrame
            df = pd.DataFrame(predictions)
            
            # Drop MongoDB ObjectId column
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"✓ Predictions exported to {filename}")
            return True
        
        except Exception as e:
            print(f"✗ Error exporting to CSV: {str(e)}")
            return False
    
    def delete_old_predictions(self, days: int = 30) -> int:
        """
        Delete predictions older than specified days
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted records
        """
        try:
            if self.collection is None:
                print("✗ MongoDB not connected")
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=days)
            result = self.collection.delete_many({"timestamp": {"$lt": cutoff_date}})
            
            print(f"✓ Deleted {result.deleted_count} old predictions")
            return result.deleted_count
        
        except Exception as e:
            print(f"✗ Error deleting old predictions: {str(e)}")
            return 0
    

    def delete_prediction_by_id(self, prediction_id: str) -> bool:
        """
        Delete a specific prediction by MongoDB ObjectId
        
        Args:
            prediction_id: MongoDB ObjectId as string
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if self.collection is None:
                print("✗ MongoDB not connected")
                return False
            
            from bson.objectid import ObjectId
            
            # Convert string to ObjectId
            try:
                obj_id = ObjectId(prediction_id)
            except:
                print(f"✗ Invalid ObjectId format: {prediction_id}")
                return False
            
            result = self.collection.delete_one({"_id": obj_id})
            
            if result.deleted_count > 0:
                print(f"✓ Prediction deleted (ID: {prediction_id})")
                return True
            else:
                print(f"✗ No prediction found with ID: {prediction_id}")
                return False
        
        except Exception as e:
            print(f"✗ Error deleting prediction: {str(e)}")
            return False
        

    def clear_all_predictions(self) -> bool:
        """
        Clear all predictions from MongoDB (use with caution!)
        
        Returns:
            True if cleared successfully
        """
        try:
            if self.collection is None:
                print("✗ MongoDB not connected")
                return False
            
            result = self.collection.delete_many({})
            print(f"✓ Deleted {result.deleted_count} predictions from database")
            return True
        
        except Exception as e:
            print(f"✗ Error clearing predictions: {str(e)}")
            return False
    
    def get_database_info(self) -> Dict:
        """Get MongoDB database information"""
        try:
            if not self.client:
                return {
                    "status": "disconnected",
                    "message": "MongoDB not connected"
                }
            
            stats = self.collection.count_documents({})
            
            return {
                "status": "connected",
                "database": self.db_name,
                "collection": self.collection_name,
                "total_documents": stats,
                "uri": self.mongodb_uri,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("✓ MongoDB connection closed")



# EXAMPLE USAGE


if __name__ == "__main__":
    # Initialize database
    init_database()
    
    db = PredictionDatabase()
    
    # Example: Save a prediction
    example_input = {
        'age': 35,
        'purchase_frequency': 20,
        'avg_order_value': 150.0,
        'num_orders': 25,
        'customer_lifetime_days': 365,
        'recency': 30,
        'frequency_score': 4
    }
    
    example_engineered = {
        'total_spending': 3750.0,
        'recency_score': 4,
        'monetary_score': 3.75,
        'rfm_combined': 11.75,
    }
    
    db.save_prediction(
        input_data=example_input,
        engineered_features=example_engineered,
        scaled_prediction=0.65,
        predicted_clv=8500.50,
        customer_segment="High Value",
        comparison_to_average=11.3,
        confidence_score=0.95
    )
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\n Database Statistics:")
    print(f"Total Predictions: {stats.get('total_predictions')}")
    print(f"Average CLV: ${stats.get('average_clv')}")
    
    # Get database info
    info = db.get_database_info()
    print(f"\n MongoDB Info:")
    print(f"Status: {info.get('status')}")
    print(f"Total Documents: {info.get('total_documents')}")
    
    db.close()