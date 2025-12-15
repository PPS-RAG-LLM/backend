"""가짜 컴퍼니 로그인 라우터"""
# routers/mock_company.py 수정
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse
import bcrypt
import jwt
import datetime
from config import config
from utils import logger

logger = logger(__name__)

mock_company_router = APIRouter(prefix="/mock-company", tags=["TEST"])

# 해시된 비밀번호로 변경
FAKE_COMPANY_EMPLOYEES = {
    "pps_admin": {"password": bcrypt.hashpw("admin1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "iju1234": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "mingue123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "rlwjd123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "ruah0807": {"password": bcrypt.hashpw("12345678".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "bum123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")}
}

@mock_company_router.post("/login")
def company_login(username: str = Form(...), password: str = Form(...)):
    """가짜 회사 로그인 처리"""
    # 1. 회사 자체 인증 (생략 - 기존 로직 유지)
    username = username.strip()
    password = password.strip()
    # 1. [복구] 아이디/비밀번호 검증 로직
    employee = FAKE_COMPANY_EMPLOYEES.get(username)

    if not employee:
        return HTMLResponse("<h3>로그인 실패: 존재하지 않는 사용자입니다.</h3>", status_code=401)

    hashed_password = employee["password"].encode("utf-8")
    if not bcrypt.checkpw(password.encode("utf-8"), hashed_password):
        return HTMLResponse("<h3>로그인 실패: 비밀번호가 틀렸습니다.</h3>", status_code=401)

    logger.info(f"✅ 회사 인증 성공: {username}")

    ## 2. [핵심] 우리 서비스용 SSO 토큰 생성 (Handshake)
    # 실제로는 이 비밀키를 회사가 안전하게 보관하고 있어야 함
    shared_secret = config.get("server").get("sso_secret_key")
    logger.info(f"shared_secret: {shared_secret}")
    payload = {
        "username": username,
        "iss": "PPS_MOCK_COMPANY",
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8) # 8시간 유효
    }

    sso_token = jwt.encode(payload, shared_secret, algorithm="HS256")
    
    # 3. 토큰 반환 (HTML 대신 JSON)
    return {
        "message": "login success",
        "sso_token": sso_token,
    }
