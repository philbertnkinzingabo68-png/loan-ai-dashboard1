from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class SinglePredictionRequest(BaseModel):
    """Schema for single loan prediction request"""
    annual_income: float = Field(..., gt=0, description="Annual income in dollars")
    debt_to_income_ratio: float = Field(..., ge=0, le=100, description="Debt to income ratio percentage")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in dollars")
    interest_rate: float = Field(..., gt=0, le=50, description="Interest rate percentage")
    gender: str = Field(..., description="Gender (Male/Female)")
    marital_status: str = Field(..., description="Marital status (Single/Married/Divorced/Widowed)")
    education_level: str = Field(..., description="Education level (High School/Bachelor's/Master's/PhD)")
    employment_status: str = Field(..., description="Employment status (Employed/Self-Employed/Unemployed)")
    loan_purpose: str = Field(..., description="Purpose of loan (Home/Car/Education/Business/Other)")
    grade_subgrade: str = Field(..., description="Loan grade and subgrade (e.g., A1, B2, C3)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "annual_income": 75000.0,
                "debt_to_income_ratio": 25.5,
                "credit_score": 720,
                "loan_amount": 25000.0,
                "interest_rate": 8.5,
                "gender": "Male",
                "marital_status": "Married",
                "education_level": "Bachelor's",
                "employment_status": "Employed",
                "loan_purpose": "Home",
                "grade_subgrade": "B2"
            }
        }

# Authentication Models
class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: int = Field(..., description="Loan approval prediction (0=Rejected, 1=Approved)")
    probability: float = Field(..., ge=0, le=1, description="Probability of loan approval")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85
            }
        }