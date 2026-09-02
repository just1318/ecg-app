"""
ECG 판독 모의 훈련 앱 - Streamlit 통합 버전 (무료/로컬 채점 + 사전 렌더링 이미지 버전)
- 1단계 UI(원문 잠금 -> 제출 후 공개, 인라인 색상 채점)
- 2단계 DB(ecg_cases_db_100.json) - 12유도 파형 PNG가 base64로 이미 포함되어 있음
- 3단계 채점: API 없이 케이스 DB에 저장된 keywords/misinterpret_triggers로
  로컬에서 텍스트 매칭 채점 (완전 무료, 인터넷 연결 불필요)

  ※ 파형 이미지를 매 요청마다 실시간으로 그리지 않고, 미리 만들어둔 PNG를
     base64로 디코드만 하기 때문에 Render 무료 플랜(약한 CPU)에서도 빠릅니다.
     새 케이스를 추가하고 싶으면 ecg_waveform_gen.py로 다시 생성해서
     waveform_png_b64 필드를 채워 넣으면 됩니다.

실행 전 준비:
  pip install -r requirements.txt
  streamlit run app.py
"""

import json
import re
import random
import base64
from pathlib import Path

import streamlit as st

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(page_title="ECG 판독 훈련", page_icon="🫀", layout="centered")

DB_PATH = Path(__file__).parent / "ecg_cases_db_100.json"

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
def grade_answer_local(case: dict, user_answer: str) -> list[dict]:
    """
    사용자 답안을 문장 단위로 나눠서, 케이스 DB의 findings(keywords,
    misinterpret_triggers)와 매칭한다.

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
        lower = sentence.lower()
        for finding in case["findings"]:
            if finding["id"] in matched:
                continue  # 이미 다른 문장에서 대응된 소견은 건너뜀 (첫 매칭 우선)

            verdict = None
            reason = None

            for trig in finding.get("misinterpret_triggers", []):
                if trig["trigger_word"].lower() in lower:
                    verdict = "오해석"
                    reason = trig["why_template"].format(trigger=trig["trigger_word"])
                    break

            if verdict is None and any(k.lower() in lower for k in finding["keywords"]):
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
# 세션 상태
# ----------------------------
def pick_random_case(exclude_id=None):
    """직전 문제와 다른 케이스를 랜덤으로 하나 뽑는다."""
    pool = [c for c in cases if c["case_id"] != exclude_id] if exclude_id else cases
    return random.choice(pool)["case_id"]


if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "results" not in st.session_state:
    st.session_state.results = None
if "case_id" not in st.session_state:
    st.session_state.case_id = pick_random_case()

# ----------------------------
# UI 본문
# ----------------------------
st.title("🫀 ECG 판독 훈련")
st.caption(f"랜덤 출제 · 전체 {len(cases)}개 케이스 중 1개")

col_a, col_b = st.columns([3, 1])
with col_b:
    if st.button("🔀 다른 문제", use_container_width=True):
        st.session_state.case_id = pick_random_case(exclude_id=st.session_state.case_id)
        st.session_state.submitted = False
        st.session_state.results = None
        st.rerun()

case = next(c for c in cases if c["case_id"] == st.session_state.case_id)
with col_a:
    st.markdown(f"**{case['case_id']}**")

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
            results = grade_answer_local(case, user_answer)
            st.session_state.results = results
            st.session_state.submitted = True
            st.session_state.last_answer = user_answer
            st.rerun()

# 결과 표시
else:
    st.markdown("**제출한 답안**")
    st.markdown(f"> {st.session_state.last_answer}")

    st.markdown("### 채점 결과")
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
        st.session_state.case_id = pick_random_case(exclude_id=st.session_state.case_id)
        st.session_state.submitted = False
        st.session_state.results = None
        st.rerun()
