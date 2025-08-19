"""
Digital Lending Accelerator - Flask API
Production-ready API for loan approval predictions

This API serves the trained ML model to provide:
- Real-time loan approval predictions
- Confidence scores and risk assessments
- Integration-ready endpoints for Salesforce
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanApprovalAPI:
    """Flask API for loan approval predictions."""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for Salesforce integration
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.imputer = None
        self.model_metadata = None
        
        # Load model and setup routes
        self.load_model()
        self.setup_routes()
        
    def load_model(self):
        """Load the trained model and preprocessing artifacts."""
        try:
            models_dir = Path("data/models")
            
            # Load the trained model
            model_path = models_dir / "production_loan_model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("✅ Model loaded successfully")
            else:
                logger.error("❌ Model file not found")
                return False
                
            # Load metadata
            metadata_path = models_dir / "production_metadata.pkl"
            if metadata_path.exists():
                self.model_metadata = joblib.load(metadata_path)
                logger.info("✅ Model metadata loaded")
            
            # Setup feature preprocessing
            self.setup_preprocessing()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def setup_preprocessing(self):
        """Setup preprocessing for incoming data."""
        # Define expected features (from our training)
        self.feature_columns = [
            'loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti',
            'fico_range_low', 'fico_range_high', 'emp_length',
            'home_ownership', 'verification_status', 'purpose',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc',
            'fico_avg', 'income_loan_ratio', 'risk_indicator'
        ]
        
        # Setup imputer
        self.imputer = SimpleImputer(strategy='median')
        
        logger.info(f"✅ Preprocessing setup for {len(self.feature_columns)} features")
    
    def preprocess_input(self, loan_data):
        """Preprocess input data for prediction."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([loan_data])
            
            # Feature engineering (same as training)
            if 'term' in df.columns:
                df['term'] = df['term'].astype(str).str.extract('(\\d+)').astype(float)
            
            if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
                df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
            
            if 'annual_inc' in df.columns and 'loan_amnt' in df.columns:
                df['income_loan_ratio'] = df['annual_inc'] / (df['loan_amnt'] + 1)
            
            if 'int_rate' in df.columns and 'dti' in df.columns:
                df['risk_indicator'] = df['int_rate'] * df['dti'] / 100
            
            # Handle categorical variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in df.columns:
                    # Simple label encoding (in production, use saved encoders)
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0  # Default value for missing features
            
            # Select only required features
            df_features = df[self.feature_columns]
            
            # Handle missing values
            df_clean = pd.DataFrame(
                self.imputer.fit_transform(df_features), 
                columns=self.feature_columns
            )
            
            return df_clean.values[0]  # Return as array for prediction
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise e
    
    def predict_loan_approval(self, loan_data):
        """Make loan approval prediction."""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(loan_data)
            
            # Make prediction
            prediction = self.model.predict([processed_data])[0]
            prediction_proba = self.model.predict_proba([processed_data])[0]
            
            # Calculate confidence and risk score
            confidence = max(prediction_proba) * 100
            risk_score = prediction_proba[1] * 100  # Probability of default
            
            # Determine approval status
            approval_status = "APPROVED" if prediction == 0 else "REJECTED"
            
            # Risk category
            if risk_score < 20:
                risk_category = "LOW"
            elif risk_score < 50:
                risk_category = "MEDIUM"
            else:
                risk_category = "HIGH"
            
            return {
                'approval_status': approval_status,
                'confidence_score': round(confidence, 2),
                'risk_score': round(risk_score, 2),
                'risk_category': risk_category,
                'prediction_value': int(prediction),
                'probability_approved': round(prediction_proba[0] * 100, 2),
                'probability_rejected': round(prediction_proba[1] * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise e
    
    def setup_routes(self):
        """Setup Flask API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """API health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'service': 'Digital Lending Accelerator API',
                'model_loaded': self.model is not None,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            })
        
        @self.app.route('/model/info', methods=['GET'])
        def model_info():
            """Get model information and metadata."""
            if not self.model:
                return jsonify({'error': 'Model not loaded'}), 500
                
            info = {
                'model_type': 'Ensemble Classifier',
                'features_count': len(self.feature_columns),
                'status': 'ready',
                'timestamp': datetime.now().isoformat()
            }
            
            if self.model_metadata:
                info.update({
                    'accuracy': self.model_metadata.get('accuracy', 'N/A'),
                    'model_version': self.model_metadata.get('model_version', '1.0'),
                    'training_date': self.model_metadata.get('created_date', 'N/A')
                })
            
            return jsonify(info)
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Main prediction endpoint for loan approval."""
            try:
                # Validate request
                if not request.is_json:
                    return jsonify({'error': 'Request must be JSON'}), 400
                
                loan_data = request.get_json()
                
                # Validate required fields
                required_fields = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 
                                 'fico_range_low', 'fico_range_high']
                
                missing_fields = [field for field in required_fields if field not in loan_data]
                if missing_fields:
                    return jsonify({
                        'error': 'Missing required fields',
                        'missing_fields': missing_fields
                    }), 400
                
                # Make prediction
                result = self.predict_loan_approval(loan_data)
                
                # Add metadata
                response = {
                    'success': True,
                    'prediction': result,
                    'model_info': {
                        'model_version': '1.0',
                        'prediction_timestamp': datetime.now().isoformat()
                    },
                    'business_logic': {
                        'automation_eligible': result['confidence_score'] > 75,
                        'manual_review_required': result['risk_score'] > 60,
                        'processing_time_ms': 'fast'
                    }
                }
                
                logger.info(f"Prediction made: {result['approval_status']} (confidence: {result['confidence_score']}%)")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error in prediction endpoint: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Prediction failed',
                    'message': str(e)
                }), 500
        
        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """Batch prediction endpoint for multiple loans."""
            try:
                if not request.is_json:
                    return jsonify({'error': 'Request must be JSON'}), 400
                
                data = request.get_json()
                
                if 'loans' not in data or not isinstance(data['loans'], list):
                    return jsonify({'error': 'Request must contain "loans" array'}), 400
                
                results = []
                for i, loan_data in enumerate(data['loans']):
                    try:
                        result = self.predict_loan_approval(loan_data)
                        results.append({
                            'loan_id': i,
                            'prediction': result,
                            'status': 'success'
                        })
                    except Exception as e:
                        results.append({
                            'loan_id': i,
                            'error': str(e),
                            'status': 'failed'
                        })
                
                return jsonify({
                    'success': True,
                    'total_loans': len(data['loans']),
                    'successful_predictions': len([r for r in results if r['status'] == 'success']),
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Batch prediction failed',
                    'message': str(e)
                }), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors."""
            return jsonify({
                'error': 'Endpoint not found',
                'available_endpoints': ['/health', '/model/info', '/predict', '/batch_predict']
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            return jsonify({
                'error': 'Internal server error',
                'message': 'Please contact API administrator'
            }), 500

def create_app():
    """Create and configure the Flask application."""
    api = LoanApprovalAPI()
    return api.app

if __name__ == '__main__':
    # Create the application
    api = LoanApprovalAPI()
    
    if api.model is None:
        print("❌ Failed to load model. Please ensure the model file exists.")
        exit(1)
    
    print("🚀 Digital Lending Accelerator API Starting...")
    print("🎯 Loan Approval Prediction Service")
    print("="*50)
    print("📡 Available Endpoints:")
    print("  GET  /health          - API health check")
    print("  GET  /model/info      - Model information")
    print("  POST /predict         - Single loan prediction")
    print("  POST /batch_predict   - Batch loan predictions")
    print("="*50)
    print("🔗 Ready for Salesforce integration!")
    
    # Start the Flask development server
    api.app.run(
        host='127.0.0.1',
        port=5000,
        debug=True
    )
