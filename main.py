from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from routers import feedback_report_router, final_evaluation_report_router, team_evaluation_router, evaluation_feedback_router, chat_router


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
    allow_origins=["*"], # url 설정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터
api_router = APIRouter()

app.include_router(feedback_report_router.router)
app.include_router(final_evaluation_report_router.router)
app.include_router(team_evaluation_router.router, prefix="/team-evaluation")
app.include_router(evaluation_feedback_router.router, prefix="/evaluation-feedback")
app.include_router(chat_router.router, prefix="/chat")

app.include_router(api_router, prefix="/api/ai")

# 헬스체크
@app.get("/health-check")
def root():
    return {"message": "SKoro-AI FastAPI is running!"}
