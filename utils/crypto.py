import os
import base64
try:
    from cryptography.fernet import Fernet
except ImportError:
    print("cryptography 모듈이 설치되지 않았습니다. 'pip install cryptography'를 실행해주세요.")
    raise

# [주의] 실제 운영 시에는 이 키를 환경변수(DB_ENCRYPTION_KEY)로 관리해야 안전합니다.
# 키가 바뀌면 기존에 암호화된 데이터를 복호화할 수 없습니다.
# 여기서는 예시로 고정된 키를 사용합니다. (Fernet.generate_key()로 생성된 값)
DEFAULT_KEY = b'Z7w1y2x3A4B5C6D7E8F9G0H1I2J3K4L5M6N7O8P9Q0R=' 

key = os.getenv("DB_ENCRYPTION_KEY")
if not key:
    # 환경변수가 없으면 기본 키 사용 (개발용)
    key = DEFAULT_KEY
else:
    # 환경변수 값이 문자열이면 bytes로 변환 필요할 수 있음
    if isinstance(key, str):
        key = key.encode()

try:
    cipher = Fernet(key)
except Exception:
    # 키 형식이 안 맞으면 새로 생성 (주의: 서버 재시작 시 데이터 유실 가능성 있음)
    cipher = Fernet(Fernet.generate_key())

def encrypt_data(text: str) -> str:
    """문자열을 암호화하여 반환"""
    if not text:
        return text
    try:
        # bytes로 변환 -> 암호화 -> 문자열로 복원
        return cipher.encrypt(text.encode()).decode()
    except Exception:
        return text

def decrypt_data(text: str) -> str:
    """암호화된 문자열을 복호화하여 반환. 실패 시(평문 등) 원본 반환"""
    if not text:
        return text
    try:
        return cipher.decrypt(text.encode()).decode()
    except Exception:
        # 복호화 실패 (예: 기존에 평문으로 저장된 데이터) 시 그대로 반환
        return text