from enum import auto
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
security = HTTPBearer(auto_error=False)

def get_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> int:
    if creds is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        payload = jwt.decode(creds.credentials, "CHANGE_ME_SECRET", algorithms=["HS256"])
        return int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid token")