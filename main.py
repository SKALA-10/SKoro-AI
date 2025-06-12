from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import feedback_report_router, final_evaluation_report_router, team_evaluation_router, evaluation_feedback_router, chat_router


app = FastAPI(
    title="SKoro-AI API",
    version="1.0.0"
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
app.include_router(feedback_report_router.router, prefix="/api")
app.include_router(final_evaluation_report_router.router, prefix="/api")
app.include_router(team_evaluation_router.router, prefix="/api/team-evaluation")
app.include_router(evaluation_feedback_router.router, prefix="/api/evaluation-feedback")
app.include_router(chat_router.router, prefix="/api/chat")

# 헬스체크
@app.get("/health-check")
def root():
    return {"message": "SKoro-AI FastAPI is running!"}
