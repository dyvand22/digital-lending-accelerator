-- Digital Lending Accelerator - Monitoring Database Schema
-- Creates tables for logging ML predictions, performance metrics, and analytics

-- Create database
CREATE DATABASE IF NOT EXISTS lending_analytics;
USE lending_analytics;

-- Table for ML prediction logs
CREATE TABLE ml_prediction_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    loan_application_id VARCHAR(50) NOT NULL,
    salesforce_record_id VARCHAR(18),
    applicant_income DECIMAL(15,2),
    credit_score INT,
    loan_amount DECIMAL(15,2),
    loan_to_income_ratio DECIMAL(5,4),
    prediction VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(5,4),
    risk_score DECIMAL(5,4),
    processing_time_ms INT,
    model_version VARCHAR(20),
    api_endpoint VARCHAR(100),
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_timestamp (created_timestamp),
    INDEX idx_prediction (prediction),
    INDEX idx_confidence_score (confidence_score),
    INDEX idx_loan_application_id (loan_application_id)
);

-- Table for model performance metrics
CREATE TABLE model_performance_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_date DATE NOT NULL,
    total_predictions INT DEFAULT 0,
    approved_predictions INT DEFAULT 0,
    rejected_predictions INT DEFAULT 0,
    manual_review_predictions INT DEFAULT 0,
    avg_confidence_score DECIMAL(5,4),
    avg_processing_time_ms INT,
    accuracy_rate DECIMAL(5,4),
    precision_rate DECIMAL(5,4),
    recall_rate DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    model_version VARCHAR(20),
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_date_version (metric_date, model_version),
    INDEX idx_metric_date (metric_date)
);

-- Table for API performance metrics
CREATE TABLE api_performance_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_hour DATETIME NOT NULL,
    total_requests INT DEFAULT 0,
    successful_requests INT DEFAULT 0,
    failed_requests INT DEFAULT 0,
    avg_response_time_ms INT,
    max_response_time_ms INT,
    min_response_time_ms INT,
    error_rate DECIMAL(5,4),
    throughput_per_hour INT,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_metric_hour (metric_hour),
    INDEX idx_metric_hour (metric_hour)
);

-- Table for business analytics
CREATE TABLE business_analytics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_date DATE NOT NULL,
    total_applications INT DEFAULT 0,
    auto_approved_applications INT DEFAULT 0,
    auto_rejected_applications INT DEFAULT 0,
    manual_review_applications INT DEFAULT 0,
    automation_rate DECIMAL(5,4),
    approval_rate DECIMAL(5,4),
    avg_loan_amount DECIMAL(15,2),
    total_loan_volume DECIMAL(18,2),
    processing_time_saved_hours DECIMAL(8,2),
    cost_savings_usd DECIMAL(12,2),
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_metric_date (metric_date),
    INDEX idx_metric_date (metric_date)
);

-- Table for error logs
CREATE TABLE error_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    error_type VARCHAR(50) NOT NULL,
    error_message TEXT,
    stack_trace TEXT,
    loan_application_id VARCHAR(50),
    api_endpoint VARCHAR(100),
    request_payload JSON,
    user_agent VARCHAR(255),
    ip_address VARCHAR(45),
    severity_level ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') DEFAULT 'MEDIUM',
    resolved BOOLEAN DEFAULT FALSE,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_timestamp TIMESTAMP NULL,
    INDEX idx_created_timestamp (created_timestamp),
    INDEX idx_error_type (error_type),
    INDEX idx_severity_level (severity_level),
    INDEX idx_resolved (resolved)
);

-- Table for audit trails
CREATE TABLE audit_trails (
    id INT AUTO_INCREMENT PRIMARY KEY,
    loan_application_id VARCHAR(50) NOT NULL,
    salesforce_record_id VARCHAR(18),
    action_type VARCHAR(50) NOT NULL,
    action_description TEXT,
    user_id VARCHAR(50),
    user_type ENUM('SYSTEM', 'USER', 'ADMIN') DEFAULT 'SYSTEM',
    before_values JSON,
    after_values JSON,
    ip_address VARCHAR(45),
    user_agent VARCHAR(255),
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_timestamp (created_timestamp),
    INDEX idx_loan_application_id (loan_application_id),
    INDEX idx_action_type (action_type),
    INDEX idx_user_id (user_id)
);

-- Create views for common analytics queries

-- Daily performance summary view
CREATE VIEW daily_performance_summary AS
SELECT 
    DATE(created_timestamp) as metric_date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN prediction = 'approved' THEN 1 ELSE 0 END) as approved_count,
    SUM(CASE WHEN prediction = 'rejected' THEN 1 ELSE 0 END) as rejected_count,
    SUM(CASE WHEN prediction = 'manual_review' THEN 1 ELSE 0 END) as manual_review_count,
    AVG(confidence_score) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(loan_amount) as avg_loan_amount,
    SUM(loan_amount) as total_loan_volume
FROM ml_prediction_logs 
GROUP BY DATE(created_timestamp)
ORDER BY metric_date DESC;

-- Hourly API performance view
CREATE VIEW hourly_api_performance AS
SELECT 
    DATE_FORMAT(created_timestamp, '%Y-%m-%d %H:00:00') as metric_hour,
    COUNT(*) as total_requests,
    SUM(CASE WHEN confidence_score > 0 THEN 1 ELSE 0 END) as successful_requests,
    SUM(CASE WHEN confidence_score IS NULL THEN 1 ELSE 0 END) as failed_requests,
    AVG(processing_time_ms) as avg_response_time,
    MAX(processing_time_ms) as max_response_time,
    MIN(processing_time_ms) as min_response_time
FROM ml_prediction_logs 
GROUP BY DATE_FORMAT(created_timestamp, '%Y-%m-%d %H:00:00')
ORDER BY metric_hour DESC;

-- Model accuracy tracking view
CREATE VIEW model_accuracy_tracking AS
SELECT 
    model_version,
    DATE(created_timestamp) as metric_date,
    COUNT(*) as total_predictions,
    AVG(confidence_score) as avg_confidence,
    STDDEV(confidence_score) as confidence_std_dev,
    SUM(CASE WHEN prediction = 'approved' AND confidence_score >= 0.8 THEN 1 ELSE 0 END) as high_confidence_approvals,
    SUM(CASE WHEN prediction = 'rejected' AND confidence_score >= 0.8 THEN 1 ELSE 0 END) as high_confidence_rejections
FROM ml_prediction_logs 
WHERE model_version IS NOT NULL
GROUP BY model_version, DATE(created_timestamp)
ORDER BY model_version, metric_date DESC;

-- Create indexes for better performance
CREATE INDEX idx_ml_logs_composite ON ml_prediction_logs (created_timestamp, prediction, confidence_score);
CREATE INDEX idx_errors_composite ON error_logs (created_timestamp, severity_level, resolved);
CREATE INDEX idx_audit_composite ON audit_trails (created_timestamp, action_type, user_type);

-- Insert sample data for testing
INSERT INTO model_performance_metrics (metric_date, total_predictions, approved_predictions, rejected_predictions, manual_review_predictions, avg_confidence_score, avg_processing_time_ms, accuracy_rate, model_version) 
VALUES 
(CURDATE(), 0, 0, 0, 0, 0.0000, 0, 0.0000, 'v1.0'),
(DATE_SUB(CURDATE(), INTERVAL 1 DAY), 150, 90, 40, 20, 0.8500, 245, 0.9200, 'v1.0'),
(DATE_SUB(CURDATE(), INTERVAL 2 DAY), 142, 85, 35, 22, 0.8300, 238, 0.9100, 'v1.0');

COMMIT;
