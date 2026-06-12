# AI Nutrition Server

음식 이미지 또는 텍스트를 입력하면 **Google Gemini**와 **Anthropic Claude**가 동시에 칼로리·영양소를 분석하는 FastAPI 서버입니다.  
두 AI의 결과를 독립적으로 반환하므로 앱에서 AI별 결과를 비교하고 사용자가 원하는 값을 선택할 수 있습니다.

---

## 기술 스택

| 항목 | 내용 |
|---|---|
| Runtime | Python 3.9+ |
| Framework | FastAPI + Uvicorn |
| AI | Google Gemini (`gemini-2.5-flash`), Anthropic Claude (`claude-sonnet-4-6`) |
| LangChain | `langchain-google-genai`, `langchain-anthropic` |
| 직렬화 | Pydantic v2 (camelCase alias → Android Gson 호환) |

---

## 프로젝트 구조

```
ai-nutrition-server/
├── main.py                        # FastAPI 앱 진입점
├── .env                           # API 키 (Git 제외)
├── .env.example                   # 환경변수 예시
└── app/
    ├── api/
    │   └── endpoints.py           # 라우터 (4개 엔드포인트)
    ├── core/
    │   └── config.py              # 환경변수 로드 (Settings)
    ├── models/
    │   └── schemas.py             # Pydantic 응답 모델
    ├── services/
    │   └── nutrition_service.py   # AI 호출·파싱·병렬 처리 로직
    └── utils/
        └── image_utils.py         # 이미지 전처리 (HEIC→JPEG 변환 등)
```

---

## 환경 설정

### 1. 패키지 설치

```bash
pip install fastapi uvicorn python-dotenv pydantic
pip install langchain-google-genai langchain-anthropic langchain-core
pip install pillow pillow-heif   # HEIC 이미지 지원 (선택)
```

### 2. 환경변수 설정

`.env` 파일을 생성하고 API 키를 입력합니다.

```env
# Google AI Studio (Gemini)
GOOGLE_API_KEY=your_google_api_key

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key

# 모델 ID (선택 — 기본값 사용 권장)
GEMINI_MODEL_ID=gemini-2.5-flash
CLAUDE_MODEL_ID=claude-sonnet-4-6

# 이미지 분석 프롬프트 (선택 — 기본값 사용 권장)
# NUTRITION_PROMPT=...
```

### 3. 서버 실행

```bash
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> **포트 충돌 시**: `kill $(lsof -ti :8000)` 으로 기존 프로세스를 종료한 후 재실행하세요.

---

## API 엔드포인트

### `GET /health`

서버 상태 확인

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

### `POST /v1/analyze-food`

음식 **이미지**를 업로드하면 Gemini + Claude가 칼로리와 영양소를 분석합니다.

| 항목 | 내용 |
|---|---|
| Method | `POST` |
| Content-Type | `multipart/form-data` |
| 지원 포맷 | JPEG, PNG, HEIC |

**요청**

```bash
curl -X POST http://localhost:8000/v1/analyze-food \
  -F "image=@food.jpg"
```

**응답**

```json
{
  "requestId": "uuid",
  "analyzedAt": "2026-06-09T00:00:00+00:00",
  "imageReceived": true,
  "imageSizeBytes": 61037,
  "message": "Gemini와 Claude 다중 AI 교차 검증이 완료되었습니다.",
  "geminiItems": [
    {
      "foodName": "삼겹살 구이",
      "confidence": 0.92,
      "portionDescription": "약 200g 기준 1인분",
      "caloriesKcal": 650.0,
      "macros": {
        "carbohydrateG": 2.0,
        "proteinG": 38.0,
        "fatG": 54.0
      }
    }
  ],
  "claudeItems": [
    {
      "foodName": "삼겹살",
      "confidence": 0.88,
      "portionDescription": "1인분 기준 (약 180g)",
      "caloriesKcal": 620.0,
      "macros": {
        "carbohydrateG": 1.0,
        "proteinG": 35.0,
        "fatG": 52.0
      }
    }
  ],
  "geminiTotalCaloriesKcal": 650.0,
  "claudeTotalCaloriesKcal": 620.0,
  "disclaimer": "본 결과는 생성형 AI의 추정치이며 의학적·영양학적 진단을 대체하지 않습니다."
}
```

---

### `GET /v1/analyze-food-text`

음식명 **텍스트**를 쿼리 파라미터로 입력하면 칼로리와 영양소를 분석합니다.

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `text` | string | ✅ | 음식명 (쉼표·공백 구분 모두 가능) |

**요청 예시**

```bash
# 쉼표 구분
curl "http://localhost:8000/v1/analyze-food-text?text=우동,새우튀김"

# 공백 구분
curl "http://localhost:8000/v1/analyze-food-text?text=우동%20새우튀김"
```

**응답** — `POST /v1/analyze-food`와 동일한 구조 (`imageReceived: false`, `imageSizeBytes: null`)

---

### `GET /v1/recommend-daily-calories`

나이·성별·체중·키를 기반으로 **하루 권장 칼로리**를 계산합니다.

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `age` | int | ✅ | 나이 (1~120) |
| `gender` | string | ✅ | 성별 (`male` / `female` 또는 `남성` / `여성`) |
| `weightKg` | float | ❌ | 몸무게 (kg) |
| `heightCm` | float | ❌ | 키 (cm) |

**요청 예시**

```bash
curl "http://localhost:8000/v1/recommend-daily-calories?age=30&gender=male&weightKg=70&heightCm=175"
```

**응답**

```json
{
  "dailyCaloriesKcal": 2183.5
}
```

---

## 응답 데이터 구조

### CalorieResult (음식 분석 공통)

| 필드 | 타입 | 설명 |
|---|---|---|
| `requestId` | string | 요청 추적용 UUID |
| `analyzedAt` | string | 분석 시각 (ISO-8601 UTC) |
| `imageReceived` | bool | 이미지 분석이면 `true`, 텍스트면 `false` |
| `imageSizeBytes` | int? | 이미지 크기 (텍스트 분석 시 `null`) |
| `message` | string | 분석 상태 메시지 |
| `geminiItems` | DetectedFoodItem[] | Gemini가 식별한 음식 목록 |
| `claudeItems` | DetectedFoodItem[] | Claude가 식별한 음식 목록 |
| `geminiTotalCaloriesKcal` | float? | Gemini 총 칼로리 (호출 실패 시 `null`) |
| `claudeTotalCaloriesKcal` | float? | Claude 총 칼로리 (호출 실패 시 `null`) |
| `disclaimer` | string | AI 추정치 안내 문구 |

### DetectedFoodItem

| 필드 | 타입 | 설명 |
|---|---|---|
| `foodName` | string | 음식 이름 |
| `confidence` | float | 신뢰도 (0.0 ~ 1.0) |
| `portionDescription` | string | 분량 설명 |
| `caloriesKcal` | float | 추정 칼로리 (kcal) |
| `macros` | Macros | 3대 영양소 |

### Macros

| 필드 | 타입 | 설명 |
|---|---|---|
| `carbohydrateG` | float? | 탄수화물 (g) |
| `proteinG` | float? | 단백질 (g) |
| `fatG` | float? | 지방 (g) |

---

## Android 연동 (Gson)

응답 필드명이 **camelCase**이므로 `@SerializedName` 없이 Gson으로 바로 파싱 가능합니다.

### Retrofit 인터페이스

```kotlin
interface NutritionApi {

    // 이미지 분석
    @Multipart
    @POST("v1/analyze-food")
    suspend fun analyzeFood(
        @Part image: MultipartBody.Part
    ): CalorieResult

    // 텍스트 분석
    @GET("v1/analyze-food-text")
    suspend fun analyzeFoodText(
        @Query("text") text: String
    ): CalorieResult

    // 하루 권장 칼로리
    @GET("v1/recommend-daily-calories")
    suspend fun recommendDailyCalories(
        @Query("age") age: Int,
        @Query("gender") gender: String,
        @Query("weightKg") weightKg: Float? = null,
        @Query("heightCm") heightCm: Float? = null
    ): DailyCalorieResult
}
```

### 데이터 클래스

```kotlin
data class CalorieResult(
    val requestId: String,
    val analyzedAt: String,
    val imageReceived: Boolean,
    val imageSizeBytes: Int?,
    val message: String,
    val geminiItems: List<DetectedFoodItem>,
    val claudeItems: List<DetectedFoodItem>,
    val geminiTotalCaloriesKcal: Float?,
    val claudeTotalCaloriesKcal: Float?,
    val disclaimer: String
)

data class DetectedFoodItem(
    val foodName: String,
    val confidence: Float,
    val portionDescription: String,
    val caloriesKcal: Float,
    val macros: Macros
)

data class Macros(
    val carbohydrateG: Float?,
    val proteinG: Float?,
    val fatG: Float?
)

data class DailyCalorieResult(
    val dailyCaloriesKcal: Float
)
```

### Base URL 설정

```kotlin
private val BASE_URL = if (Build.FINGERPRINT.contains("generic")) {
    "http://10.0.2.2:8000/"       // Android 에뮬레이터
} else {
    "http://192.168.1.152:8000/"  // 실제 기기 (PC의 LAN IP)
}
```

> PC의 LAN IP는 `ifconfig | grep "inet "` 으로 확인하세요. DHCP 환경이면 IP가 바뀔 수 있습니다.

---

## AI 동작 방식

```
앱 요청
  │
  ├─ Gemini 호출 ──┐
  │                 ├─ 각 AI 독립 분석
  └─ Claude 호출 ──┘
          │
          ▼
   geminiItems  ←── Gemini가 식별한 음식 목록
   claudeItems  ←── Claude가 식별한 음식 목록
          │
          ▼
     앱에서 AI 결과 비교 후 사용자 선택 저장
```

- 두 AI는 **병렬(asyncio.gather)** 로 동시에 호출되어 응답 시간을 최소화합니다.
- 한쪽 AI가 실패해도 나머지 AI의 결과만으로 응답합니다.
- `geminiTotalCaloriesKcal` / `claudeTotalCaloriesKcal` 은 각 AI가 **독립적으로** 계산한 값입니다. 두 값을 합산하면 이중 계산이 되므로 주의하세요.

---

## 환경변수 전체 목록

| 변수명 | 기본값 | 설명 |
|---|---|---|
| `GOOGLE_API_KEY` | — | Google AI Studio API 키 **(필수)** |
| `ANTHROPIC_API_KEY` | — | Anthropic API 키 **(필수)** |
| `GEMINI_MODEL_ID` | `gemini-2.5-flash` | Gemini 모델 ID |
| `CLAUDE_MODEL_ID` | `claude-sonnet-4-6` | Claude 모델 ID |
| `NUTRITION_PROMPT` | 내장 프롬프트 | 이미지 분석 시 AI에 전달할 프롬프트 |

---

## Swagger UI

서버 실행 후 브라우저에서 아래 주소로 전체 API 문서를 확인할 수 있습니다.

```
http://localhost:8000/docs
```
