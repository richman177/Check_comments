from fastapi import FastAPI, HTTPException  
from pydantic import BaseModel 
import joblib 
import uvicorn 

model = joblib.load("model_nb.pkl")
vectorizer = joblib.load("vec.pkl")

app = FastAPI(title="Review Classification API")


class ReviewInput(BaseModel):
    text: str


@app.post("/predict")
def predict_label(review: ReviewInput):
    try:
        text_vector = vectorizer.transform([review.text])
        prediction = model.predict(text_vector)[0]
        return {"prediction": prediction}
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
