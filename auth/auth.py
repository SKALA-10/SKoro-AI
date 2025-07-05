from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

SECRET_KEY = "U0tBTEFfU0tPUk9fU0VDUkVUX0tFWV8xMjM0NTY3ODkwUVdFUlRZ" # TODO
ALGORITHM = "HS384"

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

def admin_required(payload=Depends(verify_token)):
    if payload.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Admins only")
    return payload
