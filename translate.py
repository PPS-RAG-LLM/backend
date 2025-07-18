import openpyxl
import openai
import time, os, sys
from dotenv import load_dotenv
from openai import OpenAI
import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
if client is None:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 없습니다!")
else:
    print("OpenAI API 키 설정 완료")

# 엑셀 파일 경로
excel_path = '.docs/AnythingLLM_API_명세서_20250717.xlsx'
output_path = '.docs/AnythingLLM_API_명세서_20250717_gpt_translated.xlsx'

# 워크북 로드
wb = openpyxl.load_workbook(excel_path)
ws = wb.worksheets[0]

def gpt_translate(text):
    prompt = f"다음 영어 문장을 자연스러운 한국어로 번역해줘:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 또는 gpt-4
            messages=[
                {"role": "system", "content": "너는 전문 번역가야."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        if content is not None:
            return content.strip()
        else:
            return text
    except Exception as e:
        print(f"번역 오류: {e}")
        return text

# C열(3번째 열) 데이터 번역
for row in ws.iter_rows(min_row=2, min_col=3, max_col=3):  # min_row=2: 헤더 제외
    cell = row[0]
    if cell.value and isinstance(cell.value, str):
        print(f"번역 중: {cell.value[:30]}...")
        translated = gpt_translate(cell.value)
        cell.value = translated
        time.sleep(1.2)  # API rate limit 방지

# 번역된 워크북 저장
wb.save(output_path)
print(f"번역 완료! 저장 위치: {output_path}")