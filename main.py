from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from routers import evaluation_router, chat_router, kpi_generator_router, chatbot_summary_router

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

# 라우터
api_router = APIRouter()

api_router.include_router(evaluation_router.router, prefix="/evaluation")
api_router.include_router(chatbot_summary_router.router, prefix="/chatbot-summary")
api_router.include_router(chat_router.router, prefix="/chat")
api_router.include_router(kpi_generator_router.router, prefix="/kpi")

app.include_router(api_router, prefix="/api/ai")

# 헬스체크
@app.get("/health-check")
def root():
    return {"message": "SKoro-AI FastAPI is running!"}
