"""
ECG 판독 모의 훈련 앱 - Streamlit 통합 버전 (Gemini API 채점 버전)
- 1단계 UI(원문 잠금 -> 제출 후 공개, 인라인 색상 채점)
- 2단계 DB(ecg_cases_db_100.json) - 12유도 파형 PNG가 base64로 이미 포함되어 있음
- 3단계 채점: GEMINI_API_KEY 환경변수가 있으면 Gemini API로 의미 기반 채점,
  없으면 자동으로 로컬 키워드 매칭(완전 무료)으로 대체(fallback)

실행 전 준비:
  pip install -r requirements.txt
  (선택) Render 환경변수에 GEMINI_API_KEY 등록 -> 의미 기반 채점 활성화
  streamlit run app.py
"""

import json
import re
import random
import base64
import os
from pathlib import Path

import streamlit as st

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(page_title="ECG 판독 훈련", page_icon="🫀", layout="centered")

DB_PATH = Path(__file__).parent / "ecg_cases_db_100.json"

GEMINI_SYSTEM_PROMPT = """당신은 임상병리사 국가시험 수준의 심전도(ECG) 판독 채점관입니다.
아래 [정답지]와 [사용자 답안]을 비교하여, 정답지의 각 소견(finding)에 대해
사용자 답안이 어떻게 대응하는지 판정하세요.

각 소견(finding)마다 다음 중 하나로 판정합니다:

1. "정확일치" — 사용자가 해당 소견을 정답지와 의미적으로 동일하게 서술함
   (완전히 같은 단어일 필요는 없음. 예: "동리듬"과 "정상 동리듬 소견"은 동일 인정,
    "ST 분절 상승"과 "ST분절 상승"처럼 띄어쓰기 차이도 동일 인정)

2. "누락" — 사용자 답안에 해당 소견에 대한 언급이 전혀 없음

3. "오해석" — 사용자가 해당 소견 "자리"를 언급했지만 다음 중 하나로 잘못 서술함:
   (a) 다른 소견명/부위/유형으로 혼동 (예: 좌각차단 -> 우각차단)
   (b) 정상/비정상 판단을 반대로 함 (예: 정상 소견을 "이상 있음"으로 서술)
   이 경우 반드시 "왜 오해석인지"를 원문 대비 1문장으로 설명하세요.

각 소견마다 어느 사용자 문장이 대응되는지도 함께 표시하세요.

다음 JSON 형식으로만 출력하세요. 다른 텍스트, 코드블록 표시(```) 등은 절대 포함하지 마세요:

{
  "results": [
    {
      "finding_id": "F1",
      "finding_name": "정상 동리듬",
      "verdict": "정확일치" | "누락" | "오해석",
      "matched_sentence": "사용자 답안 중 대응 문장 (없으면 null)",
      "reason": "오해석일 경우에만: 원문 대비 왜 틀렸는지 1문장 설명 (그 외에는 null)"
    }
  ]
}
"""

# ----------------------------
# 스타일 (1단계 프로토타입 색상 규칙 그대로 이식)
# ----------------------------
st.markdown(
    """
    <style>
    .locked-card {
        filter: blur(5px);
        user-select: none;
        pointer-events: none;
        background: #FCFAF9;
        border: 1px solid #DDE3E2;
        border-radius: 4px;
        padding: 16px;
    }
    .unlocked-card {
        background: #FCFAF9;
        border: 1px solid #DDE3E2;
        border-radius: 4px;
        padding: 16px;
    }
    .case-meta { font-family: monospace; font-size: 12px; color: #5C6B73; }
    .finding-box { padding: 11px 12px; border-radius: 4px; margin-bottom: 8px; font-size: 14px; }
    .ok   { background: #EAF6EE; color: #1F8A4C; }
    .miss { background: #FBEAE8; color: #C1392B; }
    .wrong{ background: #F2ECFB; color: #6B46C1; }
    .finding-box b { display: block; margin-bottom: 4px; }
    .sub-row { color: #5C6B73; font-size: 12.5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# 데이터 로드
# ----------------------------
@st.cache_data
def load_cases():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


cases = load_cases()


@st.cache_data(show_spinner=False)
def get_waveform_image_bytes(case_id: str, waveform_png_b64: str) -> bytes:
    """미리 렌더링된 base64 PNG를 디코드만 함 (실시간 생성 없음 -> 빠름)."""
    return base64.b64decode(waveform_png_b64)

# ----------------------------
# 채점 로직 (로컬 키워드 매칭 - API 불필요, 완전 무료)
# ----------------------------
def _normalize(s: str) -> str:
    """띄어쓰기 차이로 오답 처리되지 않도록 공백을 제거하고 소문자로 통일."""
    return re.sub(r"\s+", "", s.lower())


def grade_answer_local(case: dict, user_answer: str) -> list[dict]:
    """
    사용자 답안을 문장 단위로 나눠서, 케이스 DB의 findings(keywords,
    misinterpret_triggers)와 매칭한다. 띄어쓰기 차이(예: "ST분절" vs "ST 분절")는
    같은 표현으로 인정한다.

    판정 우선순위:
    1) misinterpret_triggers의 trigger_word가 문장에 있으면 -> '오해석'
       (원래 keyword와 겹치지 않아도 됨. 예: "우심실비대"의 정답 자리에
        "좌심실비대"라고만 써도, keyword가 안 겹치지만 오해석으로 잡아야 함)
    2) trigger가 없고 keyword만 있으면 -> '정확일치'
    3) 둘 다 없으면 -> 해당 문장은 이 소견과 무관
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s*", user_answer) if s.strip()]
    matched = {}  # finding_id -> result dict

    for sentence in sentences:
        norm_sentence = _normalize(sentence)
        for finding in case["findings"]:
            if finding["id"] in matched:
                continue  # 이미 다른 문장에서 대응된 소견은 건너뜀 (첫 매칭 우선)

            verdict = None
            reason = None

            for trig in finding.get("misinterpret_triggers", []):
                if _normalize(trig["trigger_word"]) in norm_sentence:
                    verdict = "오해석"
                    reason = trig["why_template"].format(trigger=trig["trigger_word"])
                    break

            if verdict is None and any(_normalize(k) in norm_sentence for k in finding["keywords"]):
                verdict = "정확일치"

            if verdict is None:
                continue  # 이 문장은 이 소견과 무관

            matched[finding["id"]] = {
                "finding_id": finding["id"],
                "finding_name": finding["name_kr"],
                "verdict": verdict,
                "matched_sentence": sentence,
                "reason": reason,
            }

    results = []
    for finding in case["findings"]:
        if finding["id"] in matched:
            results.append(matched[finding["id"]])
        else:
            results.append({
                "finding_id": finding["id"],
                "finding_name": finding["name_kr"],
                "verdict": "누락",
                "matched_sentence": None,
                "reason": None,
            })
    return results


# ----------------------------
# 채점 로직 (Gemini API - 의미 기반, 키 있을 때만)
# ----------------------------
def grade_answer_gemini(case: dict, user_answer: str) -> list[dict]:
    """GEMINI_API_KEY가 있을 때 의미 기반으로 채점. 실패하면 예외를 던짐(호출부에서 fallback 처리)."""
    import urllib.request

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    findings_for_prompt = [
        {"id": f["id"], "name_kr": f["name_kr"], "original_phrase": f["original_phrase"]}
        for f in case["findings"]
    ]
    user_message = (
        f"[정답지]\n{json.dumps(findings_for_prompt, ensure_ascii=False, indent=2)}\n\n"
        f'[사용자 답안]\n"""\n{user_answer}\n"""'
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": GEMINI_SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)
    return parsed["results"]


def grade_answer(case: dict, user_answer: str):
    """
    GEMINI_API_KEY가 설정되어 있으면 의미 기반(Gemini) 채점을 시도하고,
    키가 없거나 호출이 실패하면 자동으로 로컬 키워드 매칭 채점으로 대체한다.
    반환값: (결과 리스트, 실제 사용된 채점 방식 "gemini" | "local")
    """
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return grade_answer_gemini(case, user_answer), "gemini"
        except Exception:
            pass  # 조용히 로컬 채점으로 대체
    return grade_answer_local(case, user_answer), "local"


# ----------------------------
# 세션 상태
# ----------------------------
def pick_random_case(exclude_id=None, difficulty=None):
    """직전 문제와 다른 케이스를, 선택된 난이도 안에서 랜덤으로 하나 뽑는다."""
    pool = cases
    if difficulty and difficulty != "전체":
        pool = [c for c in pool if c.get("difficulty") == difficulty]
    if exclude_id:
        filtered = [c for c in pool if c["case_id"] != exclude_id]
        if filtered:  # 해당 난이도에 케이스가 1개뿐이면 그대로 재사용
            pool = filtered
    return random.choice(pool)["case_id"]


DIFFICULTY_OPTIONS = ["전체", "초급", "중급", "상급"]

if "difficulty" not in st.session_state:
    st.session_state.difficulty = "전체"
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "results" not in st.session_state:
    st.session_state.results = None
if "case_id" not in st.session_state:
    st.session_state.case_id = pick_random_case(difficulty=st.session_state.difficulty)

# ----------------------------
# UI 본문
# ----------------------------
st.title("🫀 ECG 판독 훈련")

selected_difficulty = st.radio(
    "난이도",
    DIFFICULTY_OPTIONS,
    index=DIFFICULTY_OPTIONS.index(st.session_state.difficulty),
    horizontal=True,
)
if selected_difficulty != st.session_state.difficulty:
    st.session_state.difficulty = selected_difficulty
    st.session_state.case_id = pick_random_case(difficulty=selected_difficulty)
    st.session_state.submitted = False
    st.session_state.results = None
    st.rerun()

difficulty_counts = {d: sum(1 for c in cases if c.get("difficulty") == d) for d in ["초급", "중급", "상급"]}
st.caption(
    f"랜덤 출제 · {selected_difficulty} "
    f"({difficulty_counts.get(selected_difficulty, len(cases)) if selected_difficulty != '전체' else len(cases)}개 케이스 중 1개)"
)

col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("🔀 다른 문제", use_container_width=True):
        st.session_state.case_id = pick_random_case(
            exclude_id=st.session_state.case_id, difficulty=st.session_state.difficulty
        )
        st.session_state.submitted = False
        st.session_state.results = None
        st.rerun()

case = next(c for c in cases if c["case_id"] == st.session_state.case_id)
with col_a:
    st.markdown(f"**{case['case_id']}** · {case.get('difficulty', '-')}")

# 12유도 ECG 파형 (항상 표시 - 이게 판독 대상 자료)
st.markdown("**12유도 심전도**")
img_bytes = get_waveform_image_bytes(case["case_id"], case["waveform_png_b64"])
st.image(img_bytes, use_container_width=True)
st.caption("※ 실제 환자 파형이 아닌, 소견 학습용 합성(스케치) 파형입니다.")

st.markdown("&nbsp;", unsafe_allow_html=True)

# 원문 카드 (잠금 / 공개) - 채점용 정답지, 제출 전까지 숨김
st.markdown('<div class="case-meta">원문 판독지 (정답지)</div>', unsafe_allow_html=True)
if not st.session_state.submitted:
    st.markdown(
        f'<div class="locked-card">{case["original_en"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption("🔒 제출하면 원문과 한글 번역이 표시됩니다")
else:
    st.markdown(
        f'<div class="unlocked-card">'
        f'<div class="case-meta">영문 원문</div>{case["original_en"]}'
        f'<div class="case-meta" style="margin-top:12px;padding-top:12px;'
        f'border-top:1px dashed #DDE3E2;">한글 번역</div>{case["original_kr"]}'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;", unsafe_allow_html=True)

# 답안 입력 (제출 전에만 노출)
if not st.session_state.submitted:
    user_answer = st.text_area(
        "위 12유도 심전도를 보고, 판독 소견을 한글로 작성해보세요",
        height=140,
        key=f"answer_{case['case_id']}",
    )
    if st.button("제출하고 채점하기", type="primary", use_container_width=True):
        if not user_answer.strip():
            st.warning("답안을 입력해주세요.")
        else:
            results, method = grade_answer(case, user_answer)
            st.session_state.results = results
            st.session_state.grading_method = method
            st.session_state.submitted = True
            st.session_state.last_answer = user_answer
            st.rerun()

# 결과 표시
else:
    st.markdown("**제출한 답안**")
    st.markdown(f"> {st.session_state.last_answer}")

    st.markdown("### 채점 결과")
    method_label = "🧠 Gemini 의미 기반 채점" if st.session_state.get("grading_method") == "gemini" else "🔤 로컬 키워드 매칭 채점"
    st.caption(method_label)
    st.markdown(
        "🟢 정확 일치&nbsp;&nbsp;&nbsp;🟣 오해석&nbsp;&nbsp;&nbsp;🔴 누락",
        unsafe_allow_html=True,
    )

    verdict_class = {"정확일치": "ok", "누락": "miss", "오해석": "wrong"}

    for r in st.session_state.results:
        cls = verdict_class.get(r["verdict"], "miss")
        html = f'<div class="finding-box {cls}"><b>{r["verdict"]} — {r["finding_name"]}</b>'
        if r.get("matched_sentence"):
            html += f'<div class="sub-row">대응 문장: {r["matched_sentence"]}</div>'
        if r.get("reason"):
            html += f'<div class="sub-row">{r["reason"]}</div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    if st.button("다음 문제 풀기", use_container_width=True, type="primary"):
        st.session_state.case_id = pick_random_case(
            exclude_id=st.session_state.case_id, difficulty=st.session_state.difficulty
        )
        st.session_state.submitted = False
        st.session_state.results = None
        st.rerun()
