import json

# 1. openapi.json 읽기
with open("openapi.json", "r", encoding="utf-8") as f:
    openapi_content = json.load(f)

# 2. HTML 템플릿 (ReDoc 사용 - 깔끔하고 읽기 좋음)
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RUAH API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; }}
    </style>
</head>
<body>
    <!-- 1. JSON 데이터를 담을 숨겨진 div -->
    <div id="redoc-container"></div>
    
    <!-- 2. ReDoc 라이브러리 로드 (최신 안정 버전) -->
    <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>

    <script>
        // JSON 데이터를 직접 주입
        const spec = {json.dumps(openapi_content)};
        Redoc.init(spec, {{
            scrollYOffset: 50,
            hideDownloadButton: true,
            theme: {{
                colors: {{
                    primary: {{
                        main: '#32329f'
                    }}
                }}
            }}
        }}, document.getElementById('redoc-container'));
    </script>
</body>
</html>
"""

# 3. HTML 파일 저장
with open("api_docs.html", "w", encoding="utf-8") as f:
    f.write(html_template)
