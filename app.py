from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import uvicorn
import pandas as pd
import numpy as np
import io
import random
from fastapi.responses import StreamingResponse

from .model import LoanPredictor
from .database import get_db, Prediction, init_db

app = FastAPI(title="Loan Payback Prediction API")

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup_event():
    init_db()
    print("✓ Database initialized")

# ==================== MODEL ====================
try:
    predictor = LoanPredictor()
    print("✓ Predictor loaded")
except Exception as e:
    print("✗ Predictor failed:", e)
    predictor = None

# ==================== HELPERS ====================
def clean_value(value):
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value

# ==================== ROUTES ====================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/single")
def predict_single(data: dict, db: Session = Depends(get_db)):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    name = data.pop("name", "Unknown")
    
    # DEBUG: Print what data we received
    print(f"📥 Received data for: {name}")
    print(f"📥 Full data: {data}")

    required_fields = [
        "annual_income",
        "debt_to_income_ratio",
        "credit_score",
        "loan_amount",
        "interest_rate",
        "gender",
        "marital_status",
        "education_level",
        "employment_status",
        "loan_purpose",
        "grade_subgrade"
    ]

    # Prepare features and handle missing values
    features = {}
    for field in required_fields:
        value = data.get(field)
        if pd.isna(value) or value is None:
            if field in ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required numeric field '{field}' for {name}"
                )
            else:
                value = "Unknown"
        features[field] = value

    print(f"🔧 Features prepared: {features}")
    
    prediction, probability = predictor.predict_single(features)
    
    print(f"🤖 Prediction: {prediction}, Probability: {probability}")

    # Save full record in database
    try:
        print("💾 Creating Prediction object...")
        record = Prediction(
            name=name,
            annual_income=float(features["annual_income"]),
            debt_to_income_ratio=float(features["debt_to_income_ratio"]),
            credit_score=float(features["credit_score"]),
            loan_amount=float(features["loan_amount"]),
            interest_rate=float(features["interest_rate"]),
            gender=features["gender"],
            marital_status=features["marital_status"],
            education_level=features["education_level"],
            employment_status=features["employment_status"],
            loan_purpose=features["loan_purpose"],
            grade_subgrade=features["grade_subgrade"],
            prediction=int(prediction),
            probability=float(probability)
        )

        print(f"💾 Attempting to save record: {record.name}")
        
        db.add(record)
        print("✅ Record added to session")
        
        db.commit()
        print("✅ Transaction committed")
        
        db.refresh(record)
        print(f"✅ SUCCESS: Record saved with ID: {record.id}")
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "record_id": record.id  # Add this to verify
        }
        
    except Exception as e:
        print(f"❌ DATABASE ERROR: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Failed to read CSV file")

    # Debug: Print columns to confirm
    print("CSV Columns:", df.columns.tolist())
    print("First few IDs:", df['id'].head().tolist() if 'id' in df.columns else "No 'id' column found")
    
    required_fields = [
        "annual_income",
        "debt_to_income_ratio",
        "credit_score",
        "loan_amount",
        "interest_rate",
        "gender",
        "marital_status",
        "education_level",
        "employment_status",
        "loan_purpose",
        "grade_subgrade"
    ]

    results = []

    for _, row in df.iterrows():
        # Get the ID from the 'id' column
        if 'id' in df.columns and not pd.isna(row['id']):
            # Convert to string, remove decimals if it's a float
            name = str(int(row['id'])) if isinstance(row['id'], (int, float, np.integer, np.floating)) else str(row['id'])
        else:
            # Fallback if no id column
            name = f"Applicant_{_+1}"
        
        # Prepare features, fill missing values
        features = {}
        for field in required_fields:
            value = row.get(field)
            if pd.isna(value) or value is None:
                if field in ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]:
                    value = 0  # default numeric missing value
                else:
                    value = "Unknown"  # default categorical missing value
            features[field] = value

        try:
            # FIX 1: REMOVE THE ZERO-CHECK - ALWAYS TRY TO PREDICT
            # The model can handle zeros (as shown by /test-model-zero)
            prediction, probability = predictor.predict_single(features)
            
            # FIX 2: ADD DEBUG LOGGING TO SEE WHAT'S HAPPENING
            print(f"DEBUG - ID: {name}, Raw Prediction: {prediction}, Probability: {probability}")
            
            # Save to database
            db.add(Prediction(
                name=name,
                annual_income=float(features["annual_income"]),
                debt_to_income_ratio=float(features["debt_to_income_ratio"]),
                credit_score=float(features["credit_score"]),
                loan_amount=float(features["loan_amount"]),
                interest_rate=float(features["interest_rate"]),
                gender=features["gender"],
                marital_status=features["marital_status"],
                education_level=features["education_level"],
                employment_status=features["employment_status"],
                loan_purpose=features["loan_purpose"],
                grade_subgrade=features["grade_subgrade"],
                prediction=int(prediction),
                probability=float(probability)
            ))

            results.append({
                "name": name,
                "annual_income": features["annual_income"] if features["annual_income"] != 0 else "N/A",
                "credit_score": features["credit_score"] if features["credit_score"] != 0 else "N/A",
                "loan_amount": features["loan_amount"] if features["loan_amount"] != 0 else "N/A",
                "employment_status": features["employment_status"] if features["employment_status"] != "Unknown" else "N/A",
                "prediction": "Will Pay Back" if prediction == 1 else "Will Not Pay Back",
                "probability": probability
            })

        except Exception as e:
            # FIX 3: BETTER ERROR HANDLING
            error_msg = str(e)
            print(f"ERROR - ID: {name}, Error: {error_msg}")
            
            # Default to safe prediction on error
            results.append({
                "name": name,
                "annual_income": "N/A",
                "credit_score": "N/A",
                "loan_amount": "N/A",
                "employment_status": "N/A",
                "prediction": "Will Not Pay Back",  # Conservative default
                "probability": 0.0,
                "error": error_msg
            })

    db.commit()

    # FIX 4: VERIFY SUMMARY MATCHES PREDICTIONS
    will_pay_back_count = sum(1 for r in results if r.get("prediction") == "Will Pay Back")
    will_not_pay_back_count = len(results) - will_pay_back_count
    
    print(f"\n=== SUMMARY ===")
    print(f"Total applications: {len(results)}")
    print(f"Will Pay Back: {will_pay_back_count}")
    print(f"Will Not Pay Back: {will_not_pay_back_count}")
    print(f"===============\n")

    return {
        "total_applications": len(results),
        "approved_applications": will_pay_back_count,  # This should now match predictions
        "will_not_pay_back": will_not_pay_back_count,  # Added for clarity
        "results": results
    }

@app.get("/sample")
def get_sample_csv():
    # Load your dataset
    df = pd.read_csv("dataset/train.csv")
    
    # Ensure 'id' column is included
    if 'id' not in df.columns:
        # Add IDs if not present
        df.insert(0, 'id', range(1, len(df) + 1))
    
    # Pick 5 random rows
    sample_df = df.sample(n=5, random_state=random.randint(0, 1000))
    
    # Make sure 'id' column is first
    cols = ['id'] + [col for col in sample_df.columns if col != 'id']
    sample_df = sample_df[cols]
    
    # Convert to CSV in memory
    stream = io.StringIO()
    sample_df.to_csv(stream, index=False)
    stream.seek(0)
    
    # Return as downloadable CSV
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_loans.csv"}
    )


@app.get("/test-problematic-case")
async def test_problematic_case():
    if predictor is None:
        return {"error": "Model not loaded"}
    
    # CORRECTED features for ID 248644
    test_case = {
        "annual_income": 25780.85,
        "debt_to_income_ratio": 0.265,
        "credit_score": 644,
        "loan_amount": 15446.86,
        "interest_rate": 13.85,
        "gender": "Male",
        "marital_status": "Married",
        "education_level": "Master's",  # FIXED: Added apostrophe
        "employment_status": "Employed",
        "loan_purpose": "Business",
        "grade_subgrade": "D4"
    }
    
    try:
        pred, prob = predictor.predict_single(test_case)
        return {
            "test_case": test_case,
            "historical_truth": "Did NOT pay back (loan_paid_back: 0)",
            "model_prediction": "Will Pay Back" if pred == 1 else "Will Not Pay Back",
            "model_confidence": f"{prob:.1%}",
            "model_is_wrong": pred == 1,  # Model says "Will Pay Back" but truth is "Did NOT pay back"
            "note": "If model predicts 'Will Pay Back' with high confidence but historical data shows they didn't pay, there might be model accuracy issues"
        }
    except Exception as e:
        return {"error": str(e)}

# ==================== SERVE FRONTEND ====================
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# ==================== RUN ====================
if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
