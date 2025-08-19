"""
Digital Lending Accelerator - Analytics and Monitoring Service
Handles logging, metrics collection, and performance monitoring
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import mysql.connector
from mysql.connector import Error
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AnalyticsService:
    """Service for logging ML predictions and collecting analytics"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'database': os.getenv('MYSQL_DATABASE', 'lending_analytics'),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'port': int(os.getenv('MYSQL_PORT', 3306))
        }
        self.model_version = os.getenv('MODEL_VERSION', 'v1.0')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('monitoring/analytics.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self):
        """Get database connection"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except Error as e:
            self.logger.error(f"Error connecting to MySQL: {e}")
            return None
    
    def log_ml_prediction(self, 
                         loan_application_id: str,
                         input_data: Dict[str, Any],
                         prediction_result: Dict[str, Any],
                         processing_time_ms: int,
                         salesforce_record_id: Optional[str] = None) -> bool:
        """Log ML prediction to database"""
        
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            insert_query = """
                INSERT INTO ml_prediction_logs 
                (loan_application_id, salesforce_record_id, applicant_income, credit_score, 
                 loan_amount, loan_to_income_ratio, prediction, confidence_score, risk_score,
                 processing_time_ms, model_version, api_endpoint)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                loan_application_id,
                salesforce_record_id,
                input_data.get('annual_income'),
                input_data.get('credit_score'),
                input_data.get('loan_amount'),
                input_data.get('loan_to_income_ratio'),
                prediction_result.get('prediction'),
                prediction_result.get('confidence'),
                prediction_result.get('risk_score'),
                processing_time_ms,
                self.model_version,
                '/predict'
            )
            
            cursor.execute(insert_query, values)
            connection.commit()
            
            self.logger.info(f"Logged ML prediction for loan {loan_application_id}")
            return True
            
        except Error as e:
            self.logger.error(f"Error logging ML prediction: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def log_error(self, 
                  error_type: str,
                  error_message: str,
                  stack_trace: Optional[str] = None,
                  loan_application_id: Optional[str] = None,
                  api_endpoint: Optional[str] = None,
                  request_payload: Optional[Dict] = None,
                  severity_level: str = 'MEDIUM') -> bool:
        """Log error to database"""
        
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            insert_query = """
                INSERT INTO error_logs 
                (error_type, error_message, stack_trace, loan_application_id, 
                 api_endpoint, request_payload, severity_level)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                error_type,
                error_message,
                stack_trace,
                loan_application_id,
                api_endpoint,
                json.dumps(request_payload) if request_payload else None,
                severity_level
            )
            
            cursor.execute(insert_query, values)
            connection.commit()
            
            self.logger.error(f"Logged error: {error_type} - {error_message}")
            return True
            
        except Error as e:
            self.logger.error(f"Error logging error: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def log_audit_trail(self,
                       loan_application_id: str,
                       action_type: str,
                       action_description: str,
                       salesforce_record_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       user_type: str = 'SYSTEM',
                       before_values: Optional[Dict] = None,
                       after_values: Optional[Dict] = None) -> bool:
        """Log audit trail entry"""
        
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            insert_query = """
                INSERT INTO audit_trails 
                (loan_application_id, salesforce_record_id, action_type, action_description,
                 user_id, user_type, before_values, after_values)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                loan_application_id,
                salesforce_record_id,
                action_type,
                action_description,
                user_id,
                user_type,
                json.dumps(before_values) if before_values else None,
                json.dumps(after_values) if after_values else None
            )
            
            cursor.execute(insert_query, values)
            connection.commit()
            
            self.logger.info(f"Logged audit trail for loan {loan_application_id}: {action_type}")
            return True
            
        except Error as e:
            self.logger.error(f"Error logging audit trail: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def get_daily_performance_metrics(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get daily performance metrics"""
        
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            if date:
                query = "SELECT * FROM daily_performance_summary WHERE metric_date = %s"
                cursor.execute(query, (date,))
            else:
                query = "SELECT * FROM daily_performance_summary ORDER BY metric_date DESC LIMIT 1"
                cursor.execute(query)
            
            result = cursor.fetchone()
            return result
            
        except Error as e:
            self.logger.error(f"Error getting daily performance metrics: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def update_model_performance_metrics(self) -> bool:
        """Update daily model performance metrics"""
        
        connection = self.get_connection()
        if not connection:
            return False
        
        try:
            cursor = connection.cursor()
            
            # Calculate today's metrics
            today = datetime.now().date()
            
            # Get aggregated data for today
            metrics_query = """
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN prediction = 'approved' THEN 1 ELSE 0 END) as approved_predictions,
                    SUM(CASE WHEN prediction = 'rejected' THEN 1 ELSE 0 END) as rejected_predictions,
                    SUM(CASE WHEN prediction = 'manual_review' THEN 1 ELSE 0 END) as manual_review_predictions,
                    AVG(confidence_score) as avg_confidence_score,
                    AVG(processing_time_ms) as avg_processing_time_ms
                FROM ml_prediction_logs 
                WHERE DATE(created_timestamp) = %s AND model_version = %s
            """
            
            cursor.execute(metrics_query, (today, self.model_version))
            metrics = cursor.fetchone()
            
            if metrics and metrics[0] > 0:  # If there are predictions today
                # Insert or update metrics
                upsert_query = """
                    INSERT INTO model_performance_metrics 
                    (metric_date, total_predictions, approved_predictions, rejected_predictions,
                     manual_review_predictions, avg_confidence_score, avg_processing_time_ms, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    total_predictions = VALUES(total_predictions),
                    approved_predictions = VALUES(approved_predictions),
                    rejected_predictions = VALUES(rejected_predictions),
                    manual_review_predictions = VALUES(manual_review_predictions),
                    avg_confidence_score = VALUES(avg_confidence_score),
                    avg_processing_time_ms = VALUES(avg_processing_time_ms)
                """
                
                values = (
                    today,
                    metrics[0],  # total_predictions
                    metrics[1],  # approved_predictions
                    metrics[2],  # rejected_predictions
                    metrics[3],  # manual_review_predictions
                    metrics[4],  # avg_confidence_score
                    metrics[5],  # avg_processing_time_ms
                    self.model_version
                )
                
                cursor.execute(upsert_query, values)
                connection.commit()
                
                self.logger.info(f"Updated model performance metrics for {today}")
                return True
            
        except Error as e:
            self.logger.error(f"Error updating model performance metrics: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
        
        return False
    
    def get_model_accuracy_report(self, days: int = 7) -> Optional[Dict[str, Any]]:
        """Get model accuracy report for the last N days"""
        
        connection = self.get_connection()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor(dictionary=True)
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            query = """
                SELECT 
                    metric_date,
                    total_predictions,
                    approved_predictions,
                    rejected_predictions,
                    manual_review_predictions,
                    avg_confidence_score,
                    avg_processing_time_ms,
                    ROUND((approved_predictions + rejected_predictions) / total_predictions * 100, 2) as automation_rate,
                    ROUND(approved_predictions / (approved_predictions + rejected_predictions) * 100, 2) as approval_rate
                FROM model_performance_metrics 
                WHERE metric_date BETWEEN %s AND %s AND model_version = %s
                ORDER BY metric_date DESC
            """
            
            cursor.execute(query, (start_date, end_date, self.model_version))
            results = cursor.fetchall()
            
            # Calculate summary statistics
            if results:
                total_preds = sum(row['total_predictions'] for row in results)
                avg_automation_rate = sum(row['automation_rate'] or 0 for row in results) / len(results)
                avg_approval_rate = sum(row['approval_rate'] or 0 for row in results) / len(results)
                avg_confidence = sum(row['avg_confidence_score'] or 0 for row in results) / len(results)
                
                return {
                    'summary': {
                        'total_predictions': total_preds,
                        'avg_automation_rate': round(avg_automation_rate, 2),
                        'avg_approval_rate': round(avg_approval_rate, 2),
                        'avg_confidence_score': round(avg_confidence, 4),
                        'days_analyzed': len(results)
                    },
                    'daily_metrics': results
                }
            
            return None
            
        except Error as e:
            self.logger.error(f"Error getting model accuracy report: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Singleton instance
analytics_service = AnalyticsService()
