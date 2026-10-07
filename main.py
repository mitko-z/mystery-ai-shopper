from fastapi import FastAPI

app = FastAPI(title="Mystery AI Shopper")


@app.get("/")
async def root():
    """Landing message."""
    return {"message": "Mystery AI Shopper is running. See /docs."}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
