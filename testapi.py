from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from testapimodel import borrow_user
from testapidb import get_db

router = APIRouter()

@router.get("/borrow")
async def get_user(userID: str = Query(..., description = "userid"), db : Session = Depends(get_db)):
    userid = db.query(borrow_user).filter(borrow_user.user_id == userID).first()
    if not userid:
        raise HTTPException(status_code=404, detail = "user not found")
    return {"user_id" : userid.user_id, "username" : userid.username, "email" : userid.email}