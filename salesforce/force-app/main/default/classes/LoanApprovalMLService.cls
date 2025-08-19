/**
 * @description Service class for integrating with ML Loan Approval API
 * @author Digital Lending Accelerator Team
 * @date 2025-08-19
 */
public class LoanApprovalMLService {
    
    private static final String ML_API_ENDPOINT = 'http://localhost:5000/predict';
    private static final Integer TIMEOUT_MS = 10000;
    
    /**
     * @description Call ML API to get loan approval prediction
     * @param loanApplication The loan application record
     * @return MLPredictionResponse containing approval status and confidence
     */
    public static MLPredictionResponse getLoanApprovalPrediction(Loan_Application__c loanApplication) {
        
        MLPredictionResponse response = new MLPredictionResponse();
        
        try {
            // Prepare request body
            Map<String, Object> requestBody = new Map<String, Object>{
                'credit_score' => loanApplication.Credit_Score__c,
                'annual_income' => loanApplication.Applicant_Annual_Income__c,
                'loan_amount' => loanApplication.Loan_Amount__c,
                'loan_to_income_ratio' => calculateLoanToIncomeRatio(loanApplication)
            };
            
            // Make HTTP callout
            HttpRequest req = new HttpRequest();
            req.setEndpoint(ML_API_ENDPOINT);
            req.setMethod('POST');
            req.setHeader('Content-Type', 'application/json');
            req.setBody(JSON.serialize(requestBody));
            req.setTimeout(TIMEOUT_MS);
            
            Http http = new Http();
            HttpResponse httpResponse = http.send(req);
            
            // Process response
            if (httpResponse.getStatusCode() == 200) {
                Map<String, Object> responseMap = (Map<String, Object>) JSON.deserializeUntyped(httpResponse.getBody());
                
                response.isSuccess = true;
                response.prediction = (String) responseMap.get('prediction');
                response.confidence = (Decimal) responseMap.get('confidence');
                response.riskScore = (Decimal) responseMap.get('risk_score');
                
                // Determine approval status based on prediction
                if (response.prediction == 'approved') {
                    response.approvalStatus = 'Approved';
                } else if (response.prediction == 'rejected') {
                    response.approvalStatus = 'Rejected';
                } else {
                    response.approvalStatus = 'Manual Review Required';
                }
                
            } else {
                response.isSuccess = false;
                response.errorMessage = 'HTTP Error: ' + httpResponse.getStatusCode() + ' - ' + httpResponse.getBody();
            }
            
        } catch (Exception e) {
            response.isSuccess = false;
            response.errorMessage = 'Exception: ' + e.getMessage();
            System.debug(LoggingLevel.ERROR, 'ML API Error: ' + e.getMessage());
        }
        
        return response;
    }
    
    /**
     * @description Calculate loan to income ratio
     * @param loanApplication The loan application record
     * @return Decimal loan to income ratio
     */
    private static Decimal calculateLoanToIncomeRatio(Loan_Application__c loanApplication) {
        if (loanApplication.Applicant_Annual_Income__c == null || loanApplication.Applicant_Annual_Income__c == 0) {
            return 0;
        }
        return loanApplication.Loan_Amount__c / loanApplication.Applicant_Annual_Income__c;
    }
    
    /**
     * @description Wrapper class for ML API response
     */
    public class MLPredictionResponse {
        public Boolean isSuccess { get; set; }
        public String prediction { get; set; }
        public Decimal confidence { get; set; }
        public Decimal riskScore { get; set; }
        public String approvalStatus { get; set; }
        public String errorMessage { get; set; }
        
        public MLPredictionResponse() {
            this.isSuccess = false;
        }
    }
}
