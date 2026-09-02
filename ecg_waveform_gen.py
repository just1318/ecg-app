# -*- coding: utf-8 -*-
"""
12유도 ECG 파형 자동 생성기 (교육용 스케치 수준)
- 실제 환자 파형이 아니라, 소견(리듬/전도장애/축/ST-T/비대/이소성박동)을 시각적으로
  구분해서 보여주기 위한 합성(simplified) 파형입니다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
ALL_LEADS = [lead for row in LEAD_LAYOUT for lead in row]

ANTERIOR_LEADS = {"V1", "V2", "V3", "V4"}
INFERIOR_LEADS = {"II", "III", "aVF"}
LATERAL_LEADS = {"I", "aVL", "V5", "V6"}


def _gauss(t, center, width, amp):
    return amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def _make_beat_waveform(t, beat_start, pr_s, qrs_s, has_p=True, qrs_amp=1.0,
                         st_offset=0.0, t_invert=False, notch=False, wide_bizarre=False):
    """한 박동(P-QRS-T)의 파형을 t(시간축) 위에 더해서 반환"""
    y = np.zeros_like(t)

    if wide_bizarre:
        # PVC 등 이소성 박동: P파 없음, QRS 넓고 기이한 모양
        qrs_center = beat_start + 0.06
        y += _gauss(t, qrs_center - 0.03, 0.02, -0.4 * qrs_amp)
        y += _gauss(t, qrs_center, 0.03, 1.6 * qrs_amp)
        y += _gauss(t, qrs_center + 0.05, 0.03, -0.9 * qrs_amp)
        y += _gauss(t, qrs_center + 0.16, 0.05, -0.5 * qrs_amp)  # 넓고 반대 방향 T파
        return y

    p_center = beat_start
    qrs_center = beat_start + pr_s

    if has_p:
        y += _gauss(t, p_center, 0.018, 0.15)

    qrs_half = qrs_s / 2
    if notch:
        # 각차단 모양 근사: rsR' / 넓고 이중 봉우리
        y += _gauss(t, qrs_center - qrs_half * 0.5, qrs_s * 0.22, 0.9 * qrs_amp)
        y += _gauss(t, qrs_center, qrs_s * 0.18, -0.3 * qrs_amp)
        y += _gauss(t, qrs_center + qrs_half * 0.6, qrs_s * 0.25, 1.1 * qrs_amp)
    else:
        y += _gauss(t, qrs_center - qrs_half * 0.4, qrs_s * 0.22, -0.25 * qrs_amp)
        y += _gauss(t, qrs_center, qrs_s * 0.3, 1.3 * qrs_amp)
        y += _gauss(t, qrs_center + qrs_half * 0.5, qrs_s * 0.22, -0.35 * qrs_amp)

    st_center = qrs_center + qrs_half + 0.05
    y += st_offset * np.exp(-((t - st_center) ** 2) / (2 * 0.03 ** 2))

    t_center = qrs_center + qrs_half + 0.14
    t_amp = -0.3 if t_invert else 0.3
    y += _gauss(t, t_center, 0.05, t_amp)

    return y


def _lead_scale(lead, axis_key, hyper_key):
    """리드별 기본 진폭 스케일 (축/비대에 따라 조정)"""
    scale = 1.0
    if lead in ("V1", "V2"):
        scale = 0.7
    elif lead in ("V5", "V6"):
        scale = 1.0
    elif lead in ("aVR",):
        scale = -0.6  # aVR은 대부분 음성파

    if axis_key == "lad":
        if lead in ("II", "III", "aVF"):
            scale *= -0.6
        elif lead == "I":
            scale *= 1.2
    elif axis_key == "rad":
        if lead == "I":
            scale *= -0.6
        elif lead in ("II", "III", "aVF"):
            scale *= 1.2

    if hyper_key == "lvh" and lead in ("V5", "V6"):
        scale *= 1.9
    if hyper_key == "lvh" and lead in ("V1", "V2"):
        scale *= 1.6  # 깊은 S파 근사
    if hyper_key == "rvh" and lead in ("V1", "V2"):
        scale *= 1.7

    return scale


def generate_lead_signal(lead, params, duration=4.0, fs=500):
    t = np.arange(0, duration, 1 / fs)
    y = np.zeros_like(t)

    rhythm_key = params["rhythm_key"]
    rate = params["rate"]
    conduction_key = params["conduction_key"]
    pr_ms = params["pr_ms"] or 160
    qrs_ms = params["qrs_ms"] or 90
    axis_key = params["axis_key"]
    stt_key = params["stt_key"]
    mi_location = params["mi_location"]
    hyper_key = params["hyper_key"]
    ectopy_key = params["ectopy_key"]

    pr_s = pr_ms / 1000
    qrs_s = qrs_ms / 1000
    rr = 60 / rate

    is_afib = rhythm_key in ("afib",)
    is_aflutter = rhythm_key == "aflutter"
    has_p = not (is_afib or is_aflutter)
    notch = conduction_key in ("lbbb", "rbbb")

    scale = _lead_scale(lead, axis_key, hyper_key)

    st_offset = 0.0
    if stt_key == "ant_mi" and lead in ANTERIOR_LEADS:
        st_offset = 0.55
    elif stt_key == "ant_mi" and lead in INFERIOR_LEADS:
        st_offset = -0.25  # 반사성 하강
    elif stt_key == "inf_mi" and lead in INFERIOR_LEADS:
        st_offset = 0.55
    elif stt_key == "inf_mi" and lead in ANTERIOR_LEADS:
        st_offset = -0.2
    elif stt_key == "st_dep":
        st_offset = -0.25
    elif stt_key == "nonspec":
        st_offset = -0.08

    t_invert = stt_key == "t_inv" and lead in LATERAL_LEADS

    # 심방세동 baseline 잔떨림
    if is_afib:
        rng = np.random.RandomState(7)
        fib_noise = 0.04 * np.sin(2 * np.pi * 8 * t + rng.rand() * 6)
        y += fib_noise

    beat_start = 0.15
    beat_idx = 0
    pr_dynamic = pr_s
    dropped = False

    while beat_start < duration - 0.3:
        wide_bizarre = False
        skip_p = False

        # 전도장애 패턴
        if conduction_key == "avb2_1":
            # 모비츠 1형: PR이 점점 늘어나다 한 번 탈락
            cycle_pos = beat_idx % 4
            if cycle_pos == 3:
                skip_p = False
                dropped = True
            else:
                pr_dynamic = pr_s + cycle_pos * 0.05
                dropped = False
        elif conduction_key == "avb2_2":
            cycle_pos = beat_idx % 3
            dropped = cycle_pos == 2
            pr_dynamic = pr_s
        elif conduction_key == "avb3":
            # 3도 차단: P파와 QRS가 서로 무관 (단순화: QRS만 느리게, P는 별개 표시 생략)
            pr_dynamic = pr_s
        else:
            pr_dynamic = pr_s
            dropped = False

        # 이소성 박동 (PVC 이단맥 등)
        if ectopy_key in ("pvc", "pvc_bigem") and beat_idx % 2 == 1:
            wide_bizarre = True
        elif ectopy_key == "pac" and beat_idx % 3 == 2:
            pr_dynamic = pr_s * 0.6

        if is_afib or is_aflutter:
            has_p_this = False
        else:
            has_p_this = has_p

        if conduction_key == "avb3" and beat_idx % 2 == 1:
            # 방실 해리 단순화를 위해 QRS만 느리게 별도 박동으로 취급 (P는 그대로)
            pass

        if not (dropped and conduction_key in ("avb2_1", "avb2_2")):
            y += _make_beat_waveform(
                t, beat_start, pr_dynamic, qrs_s,
                has_p=has_p_this, qrs_amp=scale, st_offset=st_offset * abs(scale) if scale != 0 else st_offset,
                t_invert=t_invert, notch=notch, wide_bizarre=wide_bizarre,
            )
        else:
            # 탈락된 박동: P파만 살짝 표시
            if has_p_this:
                y += _gauss(t, beat_start, 0.018, 0.15)

        # 다음 박동 간격 (심방세동은 불규칙)
        if is_afib:
            rng2 = np.random.RandomState(beat_idx + 1)
            beat_start += rr * (0.75 + 0.5 * rng2.rand())
        else:
            beat_start += rr
        beat_idx += 1

    return t, y


def render_12_lead_png(params, out_path, title=""):
    fig, axes = plt.subplots(3, 4, figsize=(9, 5.2), facecolor="white")
    grid_color = "#F3C6C0"

    for r, row in enumerate(LEAD_LAYOUT):
        for c, lead in enumerate(row):
            ax = axes[r][c]
            t, y = generate_lead_signal(lead, params)
            ax.set_facecolor("#FFFDFB")

            for gx in np.arange(0, 4.01, 0.2):
                ax.axvline(gx, color=grid_color, linewidth=0.5, zorder=0)
            for gy in np.arange(-2, 2.01, 0.5):
                ax.axhline(gy, color=grid_color, linewidth=0.5, zorder=0)

            ax.plot(t, y, color="#16232B", linewidth=0.9, zorder=2)
            ax.set_xlim(0, 4)
            ax.set_ylim(-1.8, 1.8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#DDE3E2")
            ax.text(0.03, 1.45, lead, fontsize=10, fontweight="bold", color="#0E6E62")

    fig.suptitle(title, fontsize=11, color="#5C6B73", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import json
    d = json.load(open("/home/claude/ecg_gen/ecg_cases_db_100.json", encoding="utf-8"))
    for case in d["cases"][:3]:
        out = f"/home/claude/ecg_gen/sample_{case['case_id']}.png"
        render_12_lead_png(case["waveform_params"], out, title=case["case_id"])
        print("saved", out)
