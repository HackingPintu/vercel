from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List

# Load the trained model and scaler
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = FastAPI()

@app.get("/")
@app.head("/")
async def read_root():
    return {"message": "Hello from FastAPI!"}

class PredictRequest(BaseModel):
    entity_id: int
    lang_id: int

@app.post("/predict")
async def predict_rate(request: PredictRequest):
    results = []
    df=pd.read_excel("j2.xlsx",engine="openpyxl")
    certification=df['certificate'].unique()
    for cert in certifications:
        data = pd.DataFrame({'entity_id': [request.entity_id], 'certificate': [cert], 'lang_id': [request.lang_id]})
        data_scaled = scaler.transform(data)
        pred = best_model.predict(data_scaled)
        results.append({"certification": cert, "predicted_rate": pred[0]})
    return {"predictions": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
