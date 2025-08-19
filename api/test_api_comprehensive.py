"""
Comprehensive API Testing for Digital Lending Accelerator
Tests all endpoints with realistic loan data
"""

import requests
import json
import time
from datetime import datetime

class LoanApprovalAPITester:
    """Comprehensive API testing class."""
    
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        print("🔍 Testing Health Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Service: {data.get('service', 'N/A')}")
                print(f"   Status: {data.get('status', 'N/A')}")
                print(f"   Model Loaded: {data.get('model_loaded', 'N/A')}")
                print("   ✅ Health check passed!")
                return True
            else:
                print(f"   ❌ Health check failed!")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        print("\n🔍 Testing Model Info Endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Model Type: {data.get('model_type', 'N/A')}")
                print(f"   Features Count: {data.get('features_count', 'N/A')}")
                print(f"   Accuracy: {data.get('accuracy', 'N/A')}")
                print(f"   Model Version: {data.get('model_version', 'N/A')}")
                print("   ✅ Model info retrieved successfully!")
                return True
            else:
                print(f"   ❌ Model info request failed!")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def test_single_prediction(self):
        """Test single loan prediction."""
        print("\n🔍 Testing Single Prediction Endpoint...")
        
        # Test cases with different risk profiles
        test_cases = [
            {
                "name": "Low Risk Loan",
                "data": {
                    "loan_amnt": 10000,
                    "term": 36,
                    "int_rate": 8.5,
                    "annual_inc": 75000,
                    "dti": 12.0,
                    "fico_range_low": 750,
                    "fico_range_high": 770,
                    "emp_length": "5 years",
                    "home_ownership": "OWN",
                    "verification_status": "Verified",
                    "purpose": "debt_consolidation",
                    "delinq_2yrs": 0,
                    "inq_last_6mths": 1,
                    "open_acc": 8,
                    "pub_rec": 0,
                    "revol_bal": 5000,
                    "revol_util": 25.0,
                    "total_acc": 15
                }
            },
            {
                "name": "High Risk Loan",
                "data": {
                    "loan_amnt": 25000,
                    "term": 60,
                    "int_rate": 18.5,
                    "annual_inc": 35000,
                    "dti": 28.0,
                    "fico_range_low": 620,
                    "fico_range_high": 640,
                    "emp_length": "< 1 year",
                    "home_ownership": "RENT",
                    "verification_status": "Not Verified",
                    "purpose": "credit_card",
                    "delinq_2yrs": 2,
                    "inq_last_6mths": 4,
                    "open_acc": 12,
                    "pub_rec": 1,
                    "revol_bal": 15000,
                    "revol_util": 85.0,
                    "total_acc": 20
                }
            }
        ]
        
        for test_case in test_cases:
            print(f"\n   📊 Testing: {test_case['name']}")
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_case['data'],
                    headers={'Content-Type': 'application/json'}
                )
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get('prediction', {})
                    
                    print(f"   Approval Status: {prediction.get('approval_status', 'N/A')}")
                    print(f"   Confidence Score: {prediction.get('confidence_score', 'N/A')}%")
                    print(f"   Risk Score: {prediction.get('risk_score', 'N/A')}%")
                    print(f"   Risk Category: {prediction.get('risk_category', 'N/A')}")
                    
                    business_logic = result.get('business_logic', {})
                    print(f"   Automation Eligible: {business_logic.get('automation_eligible', 'N/A')}")
                    print(f"   Manual Review Required: {business_logic.get('manual_review_required', 'N/A')}")
                    
                    print("   ✅ Prediction successful!")
                else:
                    print(f"   ❌ Prediction failed: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        print("\n🔍 Testing Batch Prediction Endpoint...")
        
        batch_data = {
            "loans": [
                {
                    "loan_amnt": 15000,
                    "term": 36,
                    "int_rate": 10.5,
                    "annual_inc": 60000,
                    "dti": 15.0,
                    "fico_range_low": 720,
                    "fico_range_high": 740
                },
                {
                    "loan_amnt": 20000,
                    "term": 60,
                    "int_rate": 15.5,
                    "annual_inc": 45000,
                    "dti": 25.0,
                    "fico_range_low": 680,
                    "fico_range_high": 700
                },
                {
                    "loan_amnt": 12000,
                    "term": 36,
                    "int_rate": 12.0,
                    "annual_inc": 80000,
                    "dti": 10.0,
                    "fico_range_low": 780,
                    "fico_range_high": 800
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/batch_predict",
                json=batch_data,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Total Loans: {result.get('total_loans', 'N/A')}")
                print(f"   Successful Predictions: {result.get('successful_predictions', 'N/A')}")
                
                results = result.get('results', [])
                for i, loan_result in enumerate(results):
                    if loan_result.get('status') == 'success':
                        pred = loan_result.get('prediction', {})
                        print(f"   Loan {i+1}: {pred.get('approval_status', 'N/A')} "
                              f"(Confidence: {pred.get('confidence_score', 'N/A')}%)")
                    else:
                        print(f"   Loan {i+1}: Error - {loan_result.get('error', 'Unknown')}")
                
                print("   ✅ Batch prediction successful!")
                return True
            else:
                print(f"   ❌ Batch prediction failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def test_error_handling(self):
        """Test API error handling."""
        print("\n🔍 Testing Error Handling...")
        
        # Test invalid endpoint
        print("   Testing invalid endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/invalid_endpoint")
            print(f"   Status Code: {response.status_code}")
            if response.status_code == 404:
                print("   ✅ 404 handling works!")
            else:
                print("   ❌ Expected 404 status")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test invalid prediction data
        print("\n   Testing invalid prediction data...")
        try:
            invalid_data = {"invalid": "data"}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_data,
                headers={'Content-Type': 'application/json'}
            )
            print(f"   Status Code: {response.status_code}")
            if response.status_code == 400:
                print("   ✅ Input validation works!")
            else:
                print("   ❌ Expected 400 status for invalid input")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        return True
    
    def run_all_tests(self):
        """Run all API tests."""
        print("🚀 DIGITAL LENDING ACCELERATOR - API TESTING")
        print("=" * 60)
        print(f"🎯 Testing API at: {self.base_url}")
        print(f"🕐 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Model Info", self.test_model_info_endpoint),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name} Test...")
            try:
                if test_func():
                    passed += 1
                time.sleep(1)  # Small delay between tests
            except Exception as e:
                print(f"❌ Test failed with error: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 API TESTING COMPLETE!")
        print("=" * 60)
        print(f"📊 Tests Passed: {passed}/{total}")
        print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("✅ ALL TESTS PASSED! API is working correctly!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed == total

def main():
    """Main testing function."""
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)  # Give API time to start
    
    tester = LoanApprovalAPITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 API is ready for Salesforce integration!")
    else:
        print("\n🔧 API needs fixes before proceeding.")

if __name__ == "__main__":
    main()
