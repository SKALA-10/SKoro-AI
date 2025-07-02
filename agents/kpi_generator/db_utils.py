# db.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator
import os
from config.settings import DatabaseConfig

# Base 선언
Base = declarative_base()

# 설정 객체 생성
db_config = DatabaseConfig()
DATABASE_USER     = "root"
DATABASE_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_HOST     = os.getenv("DB_HOST")
DATABASE_PORT     = "3306"
DATABASE_NAME     = os.getenv("DB_NAME")
DATABASE_URL = (
    f"mysql+aiomysql://{DATABASE_USER}:{DATABASE_PASSWORD}"
    f"@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
)

# --- 1) 엔진은 한 번만 생성 ---
engine = create_async_engine(
    DATABASE_URL,
    echo=True,            # 개발 시 SQL 로그 보기
    pool_pre_ping=True,   # 쓸 때마다 ping 확인
    pool_recycle=1800,    # 1시간마다 커넥션 재생성
    pool_size=10,
    max_overflow=20,
)

# --- 2) 세션 팩토리 역시 이 엔진을 바인딩 ---
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# --- 3) FastAPI 의존성으로 세션 제공 ---
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            # only commit if everything went through persist_to_rdb
        except:
            # on any exception, immediately rollback
            await session.rollback()
            raise
        else:
            # if the endpoint completed normally (i.e. went through persist_to_rdb),
            # you’ve already committed there, so nothing more to do
            pass

# --- 4) (선택) 스키마 생성 헬퍼 ---
async def create_db_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)