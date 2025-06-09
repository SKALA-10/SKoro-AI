from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import DatabaseConfig

# 설정 객체 생성
db_config = DatabaseConfig()
DATABASE_URL = db_config.DATABASE_URL

# SQLAlchemy 엔진 생성
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 연결 확인 옵션
    echo=True  # SQL 출력 (개발 시에만 True)
)

# 세션 팩토리 생성 (ORM 쓸 경우에만 사용)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI에서 의존성 주입으로 사용할 DB 세션
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
