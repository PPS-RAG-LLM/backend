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

"""
실제 SSO 로그인 도입시 구현방법

## 1단계: 사전 협의 및 설정 (개발 전)
- 가장 먼저 양쪽 개발 담당자가 만나서 다음 정보들을 교환

### 1. 키 생성 (파트너사 수행):
- 파트너사가 스스로 개인키(Private Key)와 공개키(Public Key) 쌍을 생성합니다.
- 파트너사는 개인키를 절대 외부로 유출하지 않고 자기 서버에 깊숙이 숨깁니다.

### 2. 공개키 전달 (파트너사 → 우리):
- 파트너사가 우리에게 공개키(Public Key)를 전달합니다.
- [권장 방식 - JWKS]: 파일을 직접 주는 대신, https://partner.com/.well-known/jwks.json 같은 URL을 알려줍니다. (키가 바뀌어도 URL은 그대로라 관리가 편함)
- [단순 방식]: public_key.pem 파일을 메일이나 보안 채널로 전달합니다.

### 3. 리다이렉트 URL 등록 (우리 → 파트너사):
- 로그인 성공 후 파트너사가 사용자를 보내줄 우리 서비스 주소를 알려줍니다.
- 예: https://api.ruah.com/v1/sso/callbackd = jwt.decode(token, public_key, algorithms=["RS256"])


## 2단계: 로그인 요청 및 토큰 발급 (파트너사 개발)
사용자가 파트너사 포털에서 "NIQ 접속하기" 버튼을 눌렀을 때의 로직

### 1. 토큰 생성 (Payload 작성):
- 누가 접속하는지 정보(username, email, role 등)를 담습니다.
- 중요: aud (Audience) 필드에 "이 토큰은 Ruah 전용입니다"라는 표식을 넣습니다.
### 2. 서명 (Signing):
- 파트너사가 가진 개인키(Private Key)로 RS256 서명을 합니다.
### 3. 전송 (Redirect):
- 만들어진 JWT 토큰을 들고 사용자를 우리 서비스(https://api.ruah.com/v1/sso/callback)로 리다이렉트 시킵니다.

## 3단계: 토큰 검증 및 로그인 처리 (우리 서비스 개발)
이제 사용자가 토큰을 들고 우리 서버에 도착 이후, `service/users/session.py` 같은 곳에서 할 일.

### 1. 공개키 준비:
- 미리 받아둔 public_key.pem 파일을 로드하거나, JWKS URL에서 실시간으로 키를 조회합니다.

### 2. 서명 검증 (Verify):
- 가져온 공개키로 JWT 서명을 풉니다.
>>> jwt.decode(token, public_key, algorithms=["RS256"])
- 이 과정이 통과되면 "아, 이건 파트너사가 발급한 게 확실하구나"라고 믿을 수 있습니다.

### 3. 추가 보안 검증 (필수):
- 만료 시간(exp): 토큰이 너무 오래되지 않았는지 확인.
- 발행자(iss): 파트너사가 발급한 게 맞는지 확인.
- 수신자(aud): 다른 서비스용 토큰을 훔쳐서 우리한테 온 건 아닌지 확인.

### 4. 세션 생성:
- 검증이 끝났으니 우리 DB에 세션을 만들고 로그인 완료 처리
"""