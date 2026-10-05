from shop.api import create_app

app = create_app()


@app.get("/")
async def root():
    """Health/landing message."""
    return {"message": "Fake shop running. See /docs."}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}