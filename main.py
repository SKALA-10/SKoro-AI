from fastapi import FastAPI, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from auth.auth import admin_required
from routers import evaluation_router, chat_router, kpi_generator_router, chatbot_summary_router
from auth.auth import verify_token

app = FastAPI(
    title="SKoro-AI API",
    version="1.0.0",
    docs_url="/api/ai/docs",           
    redoc_url=None,                          
    openapi_url="/api/ai/openapi.json"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://skoro.skala25a.project.skala-ai.com",
        "http://localhost:5173",
        "http://localhost:8000"
    ], # url 설정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 ADMIN 권한 필요 API
secured_router = APIRouter(
    dependencies=[Depends(admin_required)]
)
secured_router.include_router(evaluation_router.router, prefix="/evaluation")
secured_router.include_router(kpi_generator_router.router, prefix="/kpi")
secured_router.include_router(chatbot_summary_router.router, prefix="/chatbot-summary")

# 🔓 누구나 접근 가능한 chat API
public_router = APIRouter()
public_router.include_router(chat_router.router, prefix="/chat")

# FastAPI 등록
app.include_router(secured_router, prefix="/api/ai")
app.include_router(public_router, prefix="/api/ai")

@app.get("/health-check")
def health_check():
    return {"message": "SKoro-AI FastAPI is running!"}
