import { LightningElement, track, api } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import createLoanApplication from '@salesforce/apex/LoanApplicationController.createLoanApplication';
import getLoanApprovalPrediction from '@salesforce/apex/LoanApplicationController.getLoanApprovalPrediction';

export default class LoanApplicationForm extends LightningElement {
    
    // Form data
    @track annualIncome;
    @track creditScore;
    @track loanAmount;
    @track loanToIncomeRatio = 0;
    
    // ML Results
    @track approvalStatus;
    @track confidenceScore;
    @track riskScore;
    @track showResults = false;
    
    // UI State
    @track isLoading = false;
    @track errorMessage;
    
    // Computed properties
    get isSubmitDisabled() {
        return !this.annualIncome || !this.creditScore || !this.loanAmount || this.isLoading;
    }
    
    get resultCardClass() {
        let baseClass = 'slds-box slds-theme_default slds-m-bottom_small';
        if (this.approvalStatus === 'Approved') {
            return baseClass + ' slds-theme_success';
        } else if (this.approvalStatus === 'Rejected') {
            return baseClass + ' slds-theme_error';
        } else {
            return baseClass + ' slds-theme_warning';
        }
    }
    
    get resultIcon() {
        if (this.approvalStatus === 'Approved') {
            return 'utility:success';
        } else if (this.approvalStatus === 'Rejected') {
            return 'utility:error';
        } else {
            return 'utility:warning';
        }
    }
    
    // Event handlers
    handleAnnualIncomeChange(event) {
        this.annualIncome = event.target.value;
        this.calculateLoanToIncomeRatio();
    }
    
    handleCreditScoreChange(event) {
        this.creditScore = event.target.value;
    }
    
    handleLoanAmountChange(event) {
        this.loanAmount = event.target.value;
        this.calculateLoanToIncomeRatio();
    }
    
    calculateLoanToIncomeRatio() {
        if (this.annualIncome && this.loanAmount) {
            this.loanToIncomeRatio = (this.loanAmount / this.annualIncome).toFixed(2);
        } else {
            this.loanToIncomeRatio = 0;
        }
    }
    
    async handleSubmit() {
        try {
            this.isLoading = true;
            this.errorMessage = null;
            this.showResults = false;
            
            // Validate inputs
            if (!this.validateInputs()) {
                return;
            }
            
            // Create loan application record
            const loanApplicationData = {
                Applicant_Annual_Income__c: this.annualIncome,
                Credit_Score__c: this.creditScore,
                Loan_Amount__c: this.loanAmount
            };
            
            const loanApplicationId = await createLoanApplication({ loanApplication: loanApplicationData });
            
            // Get ML prediction
            const mlResponse = await getLoanApprovalPrediction({ loanApplicationId: loanApplicationId });
            
            // Display results
            this.approvalStatus = mlResponse.approvalStatus;
            this.confidenceScore = Math.round(mlResponse.confidence * 100);
            this.riskScore = mlResponse.riskScore ? mlResponse.riskScore.toFixed(2) : null;
            this.showResults = true;
            
            // Show success toast
            this.showToast('Success', 'Loan application submitted successfully!', 'success');
            
        } catch (error) {
            console.error('Error submitting application:', error);
            this.errorMessage = 'Error submitting application: ' + (error.body?.message || error.message);
            this.showToast('Error', this.errorMessage, 'error');
        } finally {
            this.isLoading = false;
        }
    }
    
    handleReset() {
        this.annualIncome = null;
        this.creditScore = null;
        this.loanAmount = null;
        this.loanToIncomeRatio = 0;
        this.showResults = false;
        this.errorMessage = null;
        this.approvalStatus = null;
        this.confidenceScore = null;
        this.riskScore = null;
    }
    
    validateInputs() {
        let isValid = true;
        
        if (!this.annualIncome || this.annualIncome <= 0) {
            this.errorMessage = 'Please enter a valid annual income.';
            isValid = false;
        } else if (!this.creditScore || this.creditScore < 300 || this.creditScore > 850) {
            this.errorMessage = 'Please enter a valid credit score (300-850).';
            isValid = false;
        } else if (!this.loanAmount || this.loanAmount <= 0) {
            this.errorMessage = 'Please enter a valid loan amount.';
            isValid = false;
        }
        
        return isValid;
    }
    
    showToast(title, message, variant) {
        const event = new ShowToastEvent({
            title: title,
            message: message,
            variant: variant,
        });
        this.dispatchEvent(event);
    }
}
