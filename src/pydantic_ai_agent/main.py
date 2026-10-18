from fastapi import FastAPI, Depends
from pydantic_ai_agent.db_setup import User
from pydantic_ai_agent.deps import get_current_user
from pydantic_ai_agent.routers import auth, chat

app = FastAPI(
    title="E-Commerce AI Support Agent",
    description="Agentic support system backed by FastAPI and Pydantic AI",
    version="0.1.0",
)

# Auth router attach karein
app.include_router(auth.router)
app.include_router(chat.router)

@app.get("/me", tags=["users"])
async def read_current_user(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
    }