from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database Configuration for XAMPP MySQL
DB_USER = "root"  # Default XAMPP MySQL username
DB_PASSWORD = ""  # Default XAMPP MySQL password (empty)
DB_HOST = "localhost"  # Or "127.0.0.1"
DB_PORT = "3306"  # Default MySQL port
DB_NAME = "loan_prediction_db"

# MySQL Database URL
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Alternative: Use environment variables for security
# SQLALCHEMY_DATABASE_URL = os.getenv(
#     "DATABASE_URL",
#     f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# )

# Create engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,   # Recycle connections after 1 hour
    echo=False           # Set to True for SQL query logging
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# User Model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    reset_token = Column(String(255), nullable=True)
    reset_token_expiry = Column(DateTime, nullable=True)

# Predictions Model
class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), nullable=True)
    annual_income = Column(Float, nullable=False)
    debt_to_income_ratio = Column(Float, nullable=False)
    credit_score = Column(Float, nullable=False)
    loan_amount = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=False)
    gender = Column(String(50), nullable=False)
    marital_status = Column(String(50), nullable=False)
    education_level = Column(String(100), nullable=False)
    employment_status = Column(String(100), nullable=False)
    loan_purpose = Column(String(100), nullable=False)
    grade_subgrade = Column(String(10), nullable=False)
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=False)  # 0.0 to 1.0
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, nullable=True)  # Optional: link to user who made prediction

# Create all tables
def init_db():
    """Initialize database and create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables created successfully!")
    except Exception as e:
        print(f"✗ Error creating database tables: {e}")
        raise

# Dependency to get DB session
def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Test database connection
def test_connection():
    """Test if database connection is working"""
    try:
        with engine.connect() as connection:
            print("✓ Database connection successful!")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing database connection...")
    test_connection()
    print("\nInitializing database...")
    init_db()