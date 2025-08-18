# routers/mock_company.py 수정
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import requests, json, bcrypt
from datetime import datetime, timedelta
from typing import Dict, Any
from utils import logger

logger = logger(__name__)

mock_company_router = APIRouter(prefix="/mock-company", tags=["TEST"])

# 해시된 비밀번호로 변경
FAKE_COMPANY_EMPLOYEES = {
    "jongwha123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "iju1234": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "mingue123": {"password": bcrypt.hashpw("1234".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
    "rlwjd123": {"password": bcrypt.hashpw("sec123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")},
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
    
    username = username.strip()
    password = password.strip()
    
    logger.info(f"🔍 로그인 시도: username='{username}', password='{password}'")
    
    # 1. 가짜 회사 인증 (간단히)
    employee = FAKE_COMPANY_EMPLOYEES.get(username)
    if not employee:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다")
    # bcrypt로 비밀번호 검증
    hashed_password = employee["password"].encode("utf-8")
    if not bcrypt.checkpw(password.encode("utf-8"), hashed_password):
        raise HTTPException(status_code=401, detail="잘못된 비밀번호입니다")
    
    logger.info(f"✅ 회사 인증 성공: {username}")
    
    # 2. 우리 서비스에 전송할 데이터 (ID/PW만)
    sso_data = {
        "username": username,
        "password": password
    }
    
    # 3. JavaScript로 SSO 호출
    sso_data_json = json.dumps(sso_data)
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head><title>로그인 처리중...</title></head>
    <body>
        <div style="text-align: center; margin-top: 50px;">
            <h2>🔄 로그인 처리중...</h2>
            <p>{username}님, 잠시만 기다려주세요.</p>
            <div id="status">SSO 로그인 중...</div>
        </div>
        
        <script>
        async function loginToSSO() {{
            const statusDiv = document.getElementById('status');
            
            try {{
                statusDiv.textContent = 'SSO 서버에 연결 중...';
                
                const response = await fetch('/v1/sso/login', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({sso_data_json}),
                    credentials: 'include'
                }});
                
                if (response.ok) {{
                    statusDiv.textContent = '로그인 성공! 워크스페이스로 이동 중...';
                    setTimeout(() => window.location.href = '/v1/workspaces', 1000);
                }} else {{
                    const error = await response.text();
                    statusDiv.textContent = 'SSO 로그인 실패';
                    alert('SSO 로그인 실패: ' + error);
                }}
            }} catch (error) {{
                statusDiv.textContent = '연결 실패';
                alert('연결 실패: ' + error.message);
            }}
        }}
        
        window.onload = () => setTimeout(loginToSSO, 500);
        </script>
    </body>
    </html>
    """)

@mock_company_router.get("/employees")
def list_employees():
    """가짜 회사 직원 목록 (관리용)"""
    return {
        "employees": [
            {
                "employee_id": data["employee_id"],
                "name": data["full_name"],
                "department": data["dept_name"],
                "position": data["job_title"]
            }
            for data in FAKE_COMPANY_EMPLOYEES.values()  # .values() 사용
        ]
    }