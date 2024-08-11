from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
from pdfminer.high_level import extract_text
from io import BytesIO
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime


DATABASE_URL = "sqlite:///./file_logs.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FileLog(Base):
    __tablename__ = "file_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    predicted_class = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


best_model_dir = './model'
tokenizer = DistilBertTokenizer.from_pretrained(best_model_dir)
model = DistilBertForSequenceClassification.from_pretrained(best_model_dir)

app = FastAPI()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    predicted_class: str

def extract_text_from_pdf(pdf_bytes):
    pdf_file = BytesIO(pdf_bytes)
    return extract_text(pdf_file)

df = pd.read_csv('./dataset.csv')
df['class'] = df['class'].astype('category')
classes = df['class'].cat.categories

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), db: SessionLocal = Depends(get_db)):
    pdf_bytes = await file.read()
    extracted_text = extract_text_from_pdf(pdf_bytes)
    try:
        
        inputs = tokenizer(extracted_text, truncation=True, padding=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        predicted_class = classes[prediction]

        file_log = FileLog(
        filename=file.filename,
        predicted_class=str(predicted_class),
        )
        db.add(file_log)
        db.commit()
        db.refresh(file_log)
        
        return PredictResponse(predicted_class=predicted_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

