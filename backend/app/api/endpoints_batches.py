from fastapi import APIRouter, Depends, HTTPException
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from typing import List
from app.database.session import get_db
from app.database.models import Batch
from app.schemas.schemas import BatchCreate, BatchResponse

router = APIRouter()

@router.post("/", response_model=BatchResponse, status_code=201)
def create_batch(batch: BatchCreate, db: Session = Depends(get_db)):
    db_batch = db.query(Batch).filter(Batch.batch_id == batch.batch_id).first()
    if db_batch:
        raise HTTPException(status_code=400, detail="Batch ID already registered")
    
    new_batch = Batch(**batch.model_dump())
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)
    return new_batch

@router.get("/", response_model=List[BatchResponse])
def get_batches(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return db.query(Batch).offset(skip).limit(limit).all()

@router.get("/{batch_id}", response_model=BatchResponse)
def get_batch(batch_id: str, db: Session = Depends(get_db)):
    batch = db.query(Batch).filter(Batch.batch_id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch
