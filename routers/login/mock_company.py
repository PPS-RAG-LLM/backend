# routers/mock_company.py 수정
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse
import bcrypt
import jwt
import datetime
from config import config
from utils import logger
from errors import UnauthorizedError

logger = logger(__name__)

mock_company_router = APIRouter(prefix="/mock-company", tags=["TEST"])

# 해시된 비밀번호로 변경
FAKE_COMPANY_EMPLOYEES = {
    "jonghwa123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "iju1234": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "mingue123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "rlwjd123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "ruah0807": {"password": bcrypt.hashpw("12345678".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "bum123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")}
}


@mock_company_router.get("/login", response_class=HTMLResponse)
def show_company_login():
    """가짜 회사 로그인 페이지"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PPS Company 로그인</title>

        <style>
            body { font-family: Arial; margin: 50px; }
            .container { max-width: 400px; margin: 0 auto; }
            input, button { width: 100%; padding: 10px; margin: 5px 0; }
            .employee-list { background: #f5f5f5; padding: 15px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🏢 PPS Company 로그인</h2>
            
            <form action="/mock-company/login" method="post">
                <input type="text" name="username" placeholder="사용자명 (employee_id)" required>
                <input type="password" name="password" placeholder="비밀번호" required>
                <button type="submit">로그인</button>
            </form>
            
            <div class="employee-list">
                <h3>📋 테스트 계정</h3>
                <p><strong>jongwha123</strong> / 1234 (김종화 - AI 연구소 본부장)</p>
                <p><strong>iju1234</strong> / 1234 (마주이 - AI 연구소 선임 연구원)</p>
                <p><strong>mingue123</strong> / 1234 (강민규 - AI 연구소 연구원)</p>
                <p><strong>rlwjd123</strong> / sec123 (조기정 - AI 연구소 선임 연구원)</p>
                <p><strong>ruah0807</strong> / 12345678 (김루아 - AI연구소 연구원)</p>
            </div>
        </div>
    </body>
    </html>
    """
@mock_company_router.post("/login")
def company_login(username: str = Form(...), password: str = Form(...)):
    """가짜 회사 로그인 처리"""
    # 1. 회사 자체 인증 (생략 - 기존 로직 유지)
    username = username.strip()
    password = password.strip()
    
    logger.info(f"✅ 회사 인증 성공: {username}")

    ## 2. [핵심] 우리 서비스용 SSO 토큰 생성 (Handshake)
    # 실제로는 이 비밀키를 회사가 안전하게 보관하고 있어야 함
    shared_secret = config.get("server").get("sso_secret_key")
    logger
    payload = {
        "username": username,
        "iss": "PPS_MOCK_COMPANY",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5) # 5분 유효
    }
    
    sso_token = jwt.encode(payload, shared_secret, algorithm="HS256")
    
    # 3. 클라이언트에 토큰 전달 (자바스크립트가 받아서 우리 SSO API 호출)
    sso_data_js = {
        "token": sso_token
    }
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head><title>로그인 처리중...</title></head>
    <body>
        <div style="text-align: center; margin-top: 50px;">
            <h2>🔄 로그인 처리 준비 완료</h2>
            <p>사용자 {username}님. 아래 버튼을 눌러 SSO 로그인을 실행하세요.</p>
            <button id="ssoLoginBtn" style="padding:12px 20px; font-size:16px;">SSO 로그인 실행</button>
            <div id="status" style="margin-top:20px;">대기 중</div>
        </div>
        
        <script>
        async function loginToSSO() {{
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = 'SSO 서버에 연결 중...';
            try {{
                const resp = await fetch('/v1/sso/login', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({sso_data_js}),
                    credentials: 'include'
                }});
                if (resp.ok) {{
                    statusDiv.textContent = '로그인 성공! 워크스페이스로 이동 중...';
                    setTimeout(() => window.location.href = '/v1/workspaces', 1000);
                }} else {{
                    const error = await resp.text();
                    statusDiv.textContent = 'SSO 로그인 실패';
                    alert('SSO 로그인 실패: ' + error);
                }}
            }} catch (e) {{
                statusDiv.textContent = '연결 실패';
                alert('연결 실패: ' + (e && e.message ? e.message : '오류'));
            }}
        }}
        document.getElementById('ssoLoginBtn').addEventListener('click', loginToSSO);
        </script>
    </body>
    </html>
    """)
