from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime

# User Registration Schema
class UserRegister(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    full_name: Optional[str] = Field(None, max_length=200)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v

# User Login Schema
class UserLogin(BaseModel):
    username: str
    password: str

# Token Schema
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# Token Data Schema
class TokenData(BaseModel):
    username: Optional[str] = None

# User Response Schema
class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True  # For SQLAlchemy models (Pydantic v2)
        # orm_mode = True  # Use this for Pydantic v1

# User Update Schema
class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=200)

# Password Change Schema
class PasswordChange(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6, max_length=100)

# Change Password Request Schema
class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6, max_length=100)

# Password Reset Request Schema
class PasswordResetRequest(BaseModel):
    email: EmailStr

# Forgot Password Request Schema
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

# Password Reset Schema
class PasswordReset(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6, max_length=100)

# Reset Password Request Schema
class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6, max_length=100)

# Email Verification Schema
class EmailVerification(BaseModel):
    token: str

# Message Response Schema
class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None

# Login Response Schema
class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse