from fastapi import FastAPI
from app.routers.nudge import model_router
import uvicorn

app = FastAPI(
    title="Scratch & Dent Detection API",
    description="API for detecting scratches and dents in images",
    version="1.0.0"
)

app.include_router(model_router, tags=["detection"])

@app.get("/")
async def root():
    return {"message": "Scratch & Dent Detection API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

