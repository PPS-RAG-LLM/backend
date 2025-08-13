# pyright: reportUnusedImport=false
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import config
from errors import UnauthorizedError, ForbiddenError
import jwt  

security = HTTPBearer(auto_error=False)

def get_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> int:
	if creds is None:
		raise UnauthorizedError("인증이 필요합니다")
	secret = config.get("auth", {}).get("secret", "CHANGE_ME_SECRET")
	alg = config.get("auth", {}).get("algorithm", "HS256")
	try:
		payload = jwt.decode(creds.credentials, secret, algorithms=[alg])
		user_id = int(payload.get("sub"))
		return user_id
	except Exception:
		raise ForbiddenError("유효하지 않은 토큰입니다")
