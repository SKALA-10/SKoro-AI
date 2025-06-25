# ai-performance-management-system/config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig:
    DB_TYPE = os.getenv("DB_TYPE", "mariadb") # 기본값을 'mariadb'로 변경
    DB_USERNAME = os.getenv("DB_USERNAME", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD") 
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_NAME = os.getenv("DB_NAME", "skoro_db")

    @property
    def DATABASE_URL(self):
        if self.DB_PASSWORD is None:
            raise ValueError("DB_PASSWORD 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
        # f"{self.DB_TYPE}+pymysql://..." 형태로 MariaDB 드라이버를 사용
        return f"{self.DB_TYPE}+pymysql://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


if __name__ == "__main__":
    # 이 스크립트를 직접 실행할 때도 .env 파일이 로드되어야 합니다.
    # 위에서 load_dotenv()를 호출했으므로 다시 호출할 필요는 없습니다.
    db_config = DatabaseConfig()
    try:
        print(f"Generated Database URL: {db_config.DATABASE_URL}")
        print("\n참고: 운영 환경에서는 DB 비밀번호와 같은 민감 정보를 환경 변수로 설정하여 보안을 강화하는 것이 좋습니다.")
    except ValueError as e:
        print(f"DB 설정 오류: {e}")
        print("DB_PASSWORD가 .env 파일 또는 환경 변수로 올바르게 설정되었는지 확인해주세요.")