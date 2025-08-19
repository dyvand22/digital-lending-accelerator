/**
 * @description Controller class for Loan Application Lightning component
 * @author Digital Lending Accelerator Team
 * @date 2025-08-19
 */
public with sharing class LoanApplicationController {
    
    /**
     * @description Create a new loan application record
     * @param loanApplication Loan application data from the form
     * @return String ID of the created loan application
     */
    @AuraEnabled
    public static String createLoanApplication(Map<String, Object> loanApplication) {
        try {
            Loan_Application__c newLoanApp = new Loan_Application__c();
            newLoanApp.Applicant_Annual_Income__c = (Decimal) loanApplication.get('Applicant_Annual_Income__c');
            newLoanApp.Credit_Score__c = (Decimal) loanApplication.get('Credit_Score__c');
            newLoanApp.Loan_Amount__c = (Decimal) loanApplication.get('Loan_Amount__c');
            newLoanApp.ML_Approval_Status__c = 'Pending';
            
            insert newLoanApp;
            
            return newLoanApp.Id;
            
        } catch (Exception e) {
            System.debug(LoggingLevel.ERROR, 'Error creating loan application: ' + e.getMessage());
            throw new AuraHandledException('Error creating loan application: ' + e.getMessage());
        }
    }
    
    /**
     * @description Get ML approval prediction for a loan application
     * @param loanApplicationId ID of the loan application
     * @return MLPredictionWrapper containing prediction results
     */
    @AuraEnabled
    public static MLPredictionWrapper getLoanApprovalPrediction(String loanApplicationId) {
        try {
            // Retrieve the loan application
            Loan_Application__c loanApp = [
                SELECT Id, Applicant_Annual_Income__c, Credit_Score__c, Loan_Amount__c,
                       ML_Approval_Status__c, ML_Confidence_Score__c
                FROM Loan_Application__c 
                WHERE Id = :loanApplicationId 
                LIMIT 1
            ];
            
            // Call ML service
            LoanApprovalMLService.MLPredictionResponse mlResponse = 
                LoanApprovalMLService.getLoanApprovalPrediction(loanApp);
            
            // Update loan application with ML results
            if (mlResponse.isSuccess) {
                loanApp.ML_Approval_Status__c = mlResponse.approvalStatus;
                loanApp.ML_Confidence_Score__c = mlResponse.confidence;
                
                update loanApp;
                
                // Create audit log
                createAuditLog(loanApp.Id, mlResponse);
            } else {
                // Handle ML service error
                loanApp.ML_Approval_Status__c = 'Manual Review Required';
                update loanApp;
                
                System.debug(LoggingLevel.ERROR, 'ML Service Error: ' + mlResponse.errorMessage);
            }
            
            // Return wrapped response for Lightning component
            MLPredictionWrapper wrapper = new MLPredictionWrapper();
            wrapper.approvalStatus = mlResponse.isSuccess ? mlResponse.approvalStatus : 'Manual Review Required';
            wrapper.confidence = mlResponse.isSuccess ? mlResponse.confidence : 0;
            wrapper.riskScore = mlResponse.isSuccess ? mlResponse.riskScore : null;
            wrapper.errorMessage = mlResponse.isSuccess ? null : mlResponse.errorMessage;
            
            return wrapper;
            
        } catch (Exception e) {
            System.debug(LoggingLevel.ERROR, 'Error getting ML prediction: ' + e.getMessage());
            throw new AuraHandledException('Error getting ML prediction: ' + e.getMessage());
        }
    }
    
    /**
     * @description Create audit log for ML prediction
     * @param loanApplicationId ID of the loan application
     * @param mlResponse ML service response
     */
    private static void createAuditLog(String loanApplicationId, LoanApprovalMLService.MLPredictionResponse mlResponse) {
        try {
            // Create audit log record (assuming custom object exists)
            // This would be implemented based on your audit requirements
            System.debug(LoggingLevel.INFO, 'Audit Log - Loan: ' + loanApplicationId + 
                        ', Prediction: ' + mlResponse.prediction + 
                        ', Confidence: ' + mlResponse.confidence);
        } catch (Exception e) {
            System.debug(LoggingLevel.ERROR, 'Error creating audit log: ' + e.getMessage());
        }
    }
    
    /**
     * @description Wrapper class for ML prediction results to return to Lightning component
     */
    public class MLPredictionWrapper {
        @AuraEnabled public String approvalStatus { get; set; }
        @AuraEnabled public Decimal confidence { get; set; }
        @AuraEnabled public Decimal riskScore { get; set; }
        @AuraEnabled public String errorMessage { get; set; }
    }
}
