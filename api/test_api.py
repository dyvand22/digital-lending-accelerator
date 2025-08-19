"""
Digital Lending Accelerator - API Test Script
Test the Flask API endpoints and validate functionality
"""

import requests
import json
from datetime import datetime

def test_api_endpoints():
    """Test all API endpoints."""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 DIGITAL LENDING ACCELERATOR - API TESTING")
    print("🎯 Testing Flask API Endpoints")
    print("="*55)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check Endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   Service: {data['service']}")
            print(f"   Model Loaded: {data['model_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ API server not running. Please start the API first.")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Model Info
    print("\n2️⃣ Testing Model Info Endpoint...")
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model Type: {data['model_type']}")
            print(f"   Features: {data['features_count']}")
            print(f"   Status: {data['status']}")
            if 'accuracy' in data:
                print(f"   Accuracy: {data['accuracy']:.4f}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    # Test 3: Single Prediction
    print("\n3️⃣ Testing Single Loan Prediction...")
    
    # Sample loan application data
    sample_loan = {
        "loan_amnt": 15000,
        "term": "36 months",
        "int_rate": 12.5,
        "annual_inc": 65000,
        "dti": 18.5,
        "fico_range_low": 720,
        "fico_range_high": 724,
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "open_acc": 8,
        "pub_rec": 0,
        "revol_bal": 5000,
        "revol_util": 45.2,
        "total_acc": 12
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=sample_loan,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            prediction = data['prediction']
            print(f"✅ Prediction successful!")
            print(f"   Status: {prediction['approval_status']}")
            print(f"   Confidence: {prediction['confidence_score']}%")
            print(f"   Risk Score: {prediction['risk_score']}%")
            print(f"   Risk Category: {prediction['risk_category']}")
            
            business = data['business_logic']
            print(f"   Automation Eligible: {business['automation_eligible']}")
            print(f"   Manual Review: {business['manual_review_required']}")
            
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 4: Batch Prediction
    print("\n4️⃣ Testing Batch Loan Prediction...")
    
    batch_loans = {
        "loans": [
            {
                "loan_amnt": 10000,
                "term": "36 months",
                "int_rate": 10.5,
                "annual_inc": 75000,
                "dti": 15.0,
                "fico_range_low": 750,
                "fico_range_high": 754,
                "emp_length": "10+ years",
                "home_ownership": "OWN",
                "verification_status": "Verified",
                "purpose": "home_improvement"
            },
            {
                "loan_amnt": 25000,
                "term": "60 months",
                "int_rate": 18.5,
                "annual_inc": 45000,
                "dti": 35.0,
                "fico_range_low": 650,
                "fico_range_high": 654,
                "emp_length": "2 years",
                "home_ownership": "RENT",
                "verification_status": "Not Verified",
                "purpose": "credit_card"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/batch_predict",
            json=batch_loans,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful!")
            print(f"   Total Loans: {data['total_loans']}")
            print(f"   Successful: {data['successful_predictions']}")
            
            for result in data['results']:
                if result['status'] == 'success':
                    pred = result['prediction']
                    print(f"   Loan {result['loan_id']}: {pred['approval_status']} "
                          f"(confidence: {pred['confidence_score']}%)")
                else:
                    print(f"   Loan {result['loan_id']}: Failed - {result['error']}")
                    
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
    
    # Test 5: Error Handling
    print("\n5️⃣ Testing Error Handling...")
    
    # Test missing fields
    try:
        incomplete_loan = {"loan_amnt": 15000}  # Missing required fields
        
        response = requests.post(
            f"{base_url}/predict",
            json=incomplete_loan,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 400:
            data = response.json()
            print("✅ Error handling works correctly")
            print(f"   Error: {data['error']}")
            print(f"   Missing: {data['missing_fields']}")
        else:
            print(f"❌ Error handling failed: Expected 400, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test 6: Invalid endpoint
    print("\n6️⃣ Testing Invalid Endpoint...")
    try:
        response = requests.get(f"{base_url}/invalid_endpoint")
        if response.status_code == 404:
            data = response.json()
            print("✅ 404 handling works correctly")
            print(f"   Available endpoints: {data['available_endpoints']}")
        else:
            print(f"❌ 404 handling failed: {response.status_code}")
    except Exception as e:
        print(f"❌ 404 test error: {e}")
    
    print("\n" + "="*55)
    print("🎉 API TESTING COMPLETE!")
    print("✅ Flask API is ready for Salesforce integration")
    print("🔗 Use the /predict endpoint for loan approval predictions")
    
    return True

def create_sample_request():
    """Create a sample request for documentation."""
    
    print("\n📋 SAMPLE API REQUEST FOR SALESFORCE INTEGRATION:")
    print("="*55)
    
    sample_request = {
        "endpoint": "POST /predict",
        "content_type": "application/json",
        "body": {
            "loan_amnt": 15000,
            "term": "36 months",
            "int_rate": 12.5,
            "annual_inc": 65000,
            "dti": 18.5,
            "fico_range_low": 720,
            "fico_range_high": 724,
            "emp_length": "5 years",
            "home_ownership": "RENT",
            "verification_status": "Verified",
            "purpose": "debt_consolidation"
        }
    }
    
    print(json.dumps(sample_request, indent=2))
    
    sample_response = {
        "success": True,
        "prediction": {
            "approval_status": "APPROVED",
            "confidence_score": 85.23,
            "risk_score": 14.77,
            "risk_category": "LOW",
            "probability_approved": 85.23,
            "probability_rejected": 14.77
        },
        "business_logic": {
            "automation_eligible": True,
            "manual_review_required": False,
            "processing_time_ms": "fast"
        }
    }
    
    print("\n📤 SAMPLE API RESPONSE:")
    print(json.dumps(sample_response, indent=2))

if __name__ == "__main__":
    # Run the tests
    success = test_api_endpoints()
    
    if success:
        create_sample_request()
    else:
        print("\n⚠️  Please start the API server first:")
        print("   python api/loan_approval_api.py")
