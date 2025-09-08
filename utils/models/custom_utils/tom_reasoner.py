from __future__ import annotations
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import re
from openai import OpenAI

import json
from ..utils import set_gpt_model, set_openai_api_key


set_openai_api_key()


# 0) Constants & Types
SLOTS = [
    "presenting",
    "precipitating",
    "perpetuating",
    "predisposing",
    "protective",
    "impact",
]

DetectedSignals = Dict[str, List[Dict[str, Any]]]  # {slot: [{"span","start","end"}, ...]}
PPPPI = Dict[str, Dict[str, Any]]                  # {slot: {"text": str, "evidence": List[str]}}
TurnRecord = Dict[str, Any]                        # {"tom_state":..., "ppppi":..., "evidence":...}


# 1) Public entrypoint
def process_turn(
    basic_signal: Dict[str, Any],
    turn_signal: Dict[str, Any],
    raw_text: str,
    ppppi_prev: PPPPI | None = None
) -> TurnRecord:
    """One-shot processing for a user turn.

    Returns: {
        "tom_state": {...},
        "ppppi": {slot: {"text": str, "evidence": [str, ...]}},
        "evidence": {slot: [{"span","start","end"}, ...]}
    }
    """
    if ppppi_prev is None:
        ppppi_prev = init_ppppi()

    # 1) Feature engineering (delta / arousal-lite)
    features = featurize(basic_signal, turn_signal)

    # 2) Extract factual spans per slot (LLM-constrained JSON RECOMMENDED). Here: rule-based fallback
    detected = detect_ppppi_signal(raw_text)  # {"detected_signals": {slot: [...]}}
    detected_signals: DetectedSignals = detected.get("detected_signals", {slot: [] for slot in SLOTS})

    # 3) Build evidence pool from flags + detected spans + numeric signals
    ev_pool = build_evidence_pool(turn_signal, detected_signals, basic_signal)

    # 4) Decide which slots to update (internal only; not persisted)
    update_plan = update_decider(ppppi_prev, ev_pool, features)

    # 5) Summarize PPPPI (only updated slots get new text, others keep previous)
    ppppi_cur: PPPPI = {}
    for slot in SLOTS:
        ev_list = ev_pool.get(slot, [])
        if update_plan.get(slot, 0) == 1:
            text = build_ppppi_summary(slot, ev_list)
        else:
            text = ppppi_prev.get(slot, {}).get("text", "")
        ppppi_cur[slot] = {"text": text, "evidence": to_evidence_strings(ev_list)}

    # 6) ToM Reasoner state (BDI + Affect + Schemas)
    tom_state = generate_tom(features, ppppi_cur)

    # 7) Final record (no `updated` persisted)
    turn_record: TurnRecord = {
        "tom_state": tom_state,
        "ppppi": ppppi_cur
    }
    return turn_record


# 2) PPPPI init / utils
def init_ppppi() -> PPPPI:
    return {slot: {"text": "", "evidence": []} for slot in SLOTS}

# Evidence list is a list[dict]. Turn into displayable strings.

def to_evidence_strings(ev_list: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for ev in ev_list:
        s = ev.get("span", "").strip()
        if s:
            out.append(f"span:'{s}'")
    return out


# 3) Feature Engineering (delta + arousal-lite)
def featurize(basic_signal: Dict[str, Any], turn_signal: Dict[str, Any]) -> Dict[str, Any]:

    """Compute delta features and lightweight arousal indicators.


    Returns dict with:
    - delta: {emotion: float} = (turn - baseline) values for anxiety, anger, sadness, guilt, fatigue
    - A_up: overall increase in physiological/emotional arousal
    * high-arousal emotions (anxiety, anger) contribute positively
    * sadness contributes moderately (0.5 weight)
    - A_down: overall decrease in arousal (negative deltas)
    * mirrors A_up but for reductions
    - control: baseline perceived control (0-1)
    - discomfort: baseline discomfort/distress (0-1)
    - importance: baseline importance/motivation (0-1)
    - domain: problem domain (e.g., 인간관계, 학업)
    - flags: cognitive distortion flags detected in this turn
    """

    """Compute deltas and light arousal indicators. Assumes 0-1 scales."""
    aff_b = basic_signal.get("affect_baseline", {})
    aff_t = (turn_signal.get("affect_text") or {})

    def get_b(name: str) -> float:
        return float(aff_b.get(name, 0.0))

    def get_t(name_t: str) -> float:
        return float(aff_t.get(name_t, 0.0))

    # map names (baseline keys vs turn keys)
    mapping = {
        "anxiety": ("anxiety", "anxiety_t"),
        "anger": ("anger", "anger_t"),
        "sadness": ("sadness", "sadness_t"),
        "guilt": ("guilt", "guilt_t"),
        "fatigue": ("fatigue", "fatigue_t"),
    }
    delta: Dict[str, float] = {}
    for k, (bk, tk) in mapping.items():
        delta[k] = round(get_t(tk) - get_b(bk), 6)

    # lightweight arousal roll-ups (weights kept simple; can be learned later)
    high = [max(delta.get("anxiety", 0.0), 0.0), max(delta.get("anger", 0.0), 0.0)]
    low  = [max(delta.get("sadness", 0.0), 0.0)]
    A_up = sum(high) + 0.5 * sum(low)
    high_d = [max(-delta.get("anxiety", 0.0), 0.0), max(-delta.get("anger", 0.0), 0.0)]
    low_d  = [max(-delta.get("sadness", 0.0), 0.0)]
    A_down = sum(high_d) + 0.5 * sum(low_d)

    features = {
        "delta": delta,
        "A_up": round(A_up, 6),
        "A_down": round(A_down, 6),
        "control": float(basic_signal.get("control_baseline", 0.0)),
        "discomfort": float(basic_signal.get("discomfort_baseline", 0.0)),
        "importance": float(basic_signal.get("importance_baseline", 0.0)),
        "domain": basic_signal.get("domain"),
        "flags": (turn_signal.get("cognitive_flags") or {}),
    }
    return features


# 4) LLM Detector (constrained JSON) — here: rule-based fallback
def _find_spans(pat: re.Pattern, text: str) -> List[Tuple[str, int, int]]:
    spans = []
    for m in pat.finditer(text):
        s, e = m.span()
        spans.append((text[s:e], s, e))
    return spans


def detect_ppppi_signal(raw_text: str) -> Dict[str, Any]:
    """Call GPT API with constrained prompt to extract factual spans per PPPPI slot."""
    sys = (
        "너는 상담 맥락의 '증거 추출기'다. "
        "입력 텍스트에서 PPPPI 슬롯별로 **사실적 스팬**만 추출한다. "
        "추론/해석/진단/요약 금지, **입력에 존재하는 문자열 일부만** 반환한다. "
        "반드시 JSON **한 개**만 출력한다(코드블록/설명/라벨 금지)."
    )

    usr = '''
    [지시]
    다음 규칙에 따라 **문자 단위 오프셋(start/end, 0-based, end exclusive)**과 함께 스팬을 추출하라.

    [PPPPI 슬롯 정의]
    - presenting: 현재 호소 문제/감정 표현(예: 힘들다, 불안, 우울, 괴롭다, 제/내 탓 등).
    - precipitating: 최근 촉발 요인/사건/시간 단서(예: 어제/오늘/최근/이번, 말다툼/갈등/혼났다 등).
    - perpetuating: 문제를 유지·악화시키는 반복/인지 왜곡 단서(예: 항상/늘/계속, 파국적 해석, 자기비난·과잉일반화·당위문).
    - predisposing: 과거부터 이어진 취약성/배경 단서(오래된 경향/특성/환경). **근거 없으면 비워둔다.**
    - protective: 보호 요인/강점/지지(도움을 요청/받음, 가치/의미/강점 언급, 지지적 사람·자원).
    - impact: 기능 영향(수면/학업·업무/대인 기능 저하·회피·피로 등 **구체 영향** 표현).

    [추출 규칙]
    - **입력에 존재하는 원문 스팬만** 추출한다(부분 문자열). 임의 단어 생성 금지.
    - 각 슬롯 **최대 5개**까지, **가장 정보량 높은** 스팬을 우선.
    - **중복/완전포함 스팬 제거**: 같은 의미면 더 긴 한 개만 남긴다.
    - 불필요한 공백은 유지하되, **앞뒤 공백은 제외**하여 offset을 정확히 맞춘다.
    - 겹치는 스팬이 다른 슬롯 의미에 더 적합하면 **그 슬롯에만** 둔다(중복 슬롯 배치 금지).
    - 판단 불가/근거 없음인 슬롯은 **빈 배열**로 둔다.

    [출력 스키마(JSON만)]
    {
    "detected_signals": {
        "presenting":     [ {"span": str, "start": int, "end": int}, ... ],
        "precipitating":  [ {"span": str, "start": int, "end": int}, ... ],
        "perpetuating":   [ {"span": str, "start": int, "end": int}, ... ],
        "predisposing":   [ {"span": str, "start": int, "end": int}, ... ],
        "protective":     [ {"span": str, "start": int, "end": int}, ... ],
        "impact":         [ {"span": str, "start": int, "end": int}, ... ]
    }
    }

    [예시1]
    TEXT: "요즘 친구들이 날 피하는 것 같아 너무 힘들다. 어제도 말다툼이 있었고, 항상 내 탓이라 느낀다. 잠도 4시간밖에 못 잔다."
    → 정답(예시):
    {
    "detected_signals": {
        "presenting":    [ {"span":"너무 힘들다","start":16,"end":22} ],
        "precipitating": [ {"span":"어제도 말다툼","start":24,"end":31} ],
        "perpetuating":  [ {"span":"항상 내 탓","start":36,"end":42} ],
        "predisposing":  [],
        "protective":    [],
        "impact":        [ {"span":"잠도 4시간밖에 못 잔다","start":45,"end":64} ]
    }
    }

    [예시2]
    TEXT: "도와달라고 했고 상담을 받으려 한다. 최근 계속 버림받는 느낌이다."
    → 정답(예시):
    {
    "detected_signals": {
        "presenting":    [ {"span":"버림받는 느낌","start":20,"end":28} ],
        "precipitating": [ {"span":"최근","start":15,"end":17} ],
        "perpetuating":  [ {"span":"계속","start":18,"end":20} ],
        "predisposing":  [],
        "protective":    [ {"span":"도와달라고 했고 상담을 받으려 한다","start":0,"end":17} ],
        "impact":        []
    }
    }


    [출력]
    JSON 한 개만 출력하라.

    [입력 TEXT]\n
    '''
    usr+=raw_text

    model = set_gpt_model()
    client = OpenAI()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a JSON-only extractor."},
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},],
        temperature=0,)

    try:
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        return data
    except Exception as e:
        # fallback: return empty schema
        return {"detected_signals": {slot: [] for slot in SLOTS}}


# 5) Evidence Pool Builder
def build_evidence_pool(
    turn_signal: Dict[str, Any],
    detected_signals: DetectedSignals,
    basic_signal: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Combine cognitive flags + detected spans + salient numerics per slot.
    Output per slot: list of dicts with at least {"span","start","end"} or {"span"}.
    """
    pool: Dict[str, List[Dict[str, Any]]] = {slot: [] for slot in SLOTS}

    # 5.1 detected spans → pool
    for slot in SLOTS:
        for ev in detected_signals.get(slot, []):
            pool[slot].append({"span": ev.get("span", ""), "start": ev.get("start", -1), "end": ev.get("end", -1)})

    # 5.2 cognitive flags (4 kinds) → pool (as text spans if available)
    flags = (turn_signal.get("cognitive_flags") or {})
    for flag_name, info in flags.items():
        if not info:
            continue
        if info.get("present"):
            spans = info.get("spans", []) or []
            if spans:
                for sp in spans:
                    pool["presenting"].append({"span": sp.get("text", f"{flag_name} present")})
                    # also feed to precipitating/perpetuating heuristically
                    pool["perpetuating"].append({"span": sp.get("text", f"{flag_name} present")})
            else:
                pool["presenting"].append({"span": f"{flag_name} present"})


    return pool


def update_decider(
    ppppi_prev: PPPPI,
    ev_pool: Dict[str, List[Dict[str, Any]]],
    features: Dict[str, Any]
) -> Dict[str, int]:
    """
    Decide which PPPPI slots to (re)generate *this turn*.

    Baselines (control, importance, discomfort) are survey-based and static.
    -> Use them only to seed an initial hypothesis when a slot has no text yet.
    -> After that, update only when NEW textual evidence (spans) appears.
    """
    plan: Dict[str, int] = {slot: 0 for slot in SLOTS}

    def has_new(slot: str) -> bool:
        prev_evs = set(ppppi_prev.get(slot, {}).get("evidence", []))
        cur_evs = set(to_evidence_strings(ev_pool.get(slot, [])))
        return bool(cur_evs - prev_evs)

    def is_unset(slot: str) -> bool:
        return not (ppppi_prev.get(slot, {}) or {}).get("text")

    # Presenting — new spans OR (slot empty & discomfort baseline present) → seed once
    if has_new("presenting") or (
        is_unset("presenting")
        and any(ev.get("span", "").startswith("discomfort=") for ev in ev_pool.get("presenting", []))
    ):
        plan["presenting"] = 1

    # Precipitating — new precipitating spans only
    if has_new("precipitating"):
        plan["precipitating"] = 1

    # Perpetuating — new repetition/pattern spans only
    if has_new("perpetuating"):
        plan["perpetuating"] = 1

    # Predisposing — rare; new history/trait spans only
    if has_new("predisposing"):
        plan["predisposing"] = 1

    # Protective — new support/value/help spans OR (slot empty & importance baseline) → seed once
    if has_new("protective") or (
        is_unset("protective")
        and any(ev.get("span", "").startswith("importance=") for ev in ev_pool.get("protective", []))
    ):
        plan["protective"] = 1

    # Impact — new functional/impact spans OR (slot empty & control baseline) → seed once
    if has_new("impact") or (
        is_unset("impact")
        and any(ev.get("span", "").startswith("control=") for ev in ev_pool.get("impact", []))
    ):
        plan["impact"] = 1

    return plan


def build_ppppi_summary(slot: str, ev_list: List[Dict[str, Any]]) -> str:
    """PPPPI 슬롯 요약 생성기.

    - OpenAI Chat Completions API를 이용해 한 줄 한국어 요약을 생성.
    - API를 사용할 수 없으면 증거 스팬을 단순히 합쳐서 반환.
    - 제약:
      * 증거(evidence) 외 단어/사실 추가 금지
      * 한 문장 한국어 (약 35자 내외)
      * 가능성/추정 표현 최소화
      * 증거 단어 최소 1개 포함
    """
    if not ev_list:
        return ""

    # 증거 스팬 준비 (중복 제거, 순서 보존)
    spans = [ev.get("span", "").strip() for ev in ev_list if ev.get("span")]
    uniq_spans = list(dict.fromkeys([s for s in spans if s]))
    evidence_str = "; ".join(uniq_spans)

    # LLM 호출 시도
    try:
        model = set_gpt_model()
        client = OpenAI()

        sys = (
        "너는 상담 맥락의 PPPPI 요약기다. 반드시 '증거 스팬'만 사용해 한 문장으로 요약하라. "
        "추론/과장/새 정보 금지. 한국어 한 문장(≈20~35자). "
        "라벨/따옴표/괄호/마침표/이모지 금지. "
        "증거의 핵심 단어(어근/형태 변형 허용)를 1개 이상 포함할 것. "
        "빈 evidence면 빈 문자열로만 출력."
        )

        usr = f"""
        [슬롯] {slot}

        [슬롯 정의]
        - presenting: 내담자가 현재 호소하는 핵심 문제/감정(지금 무엇이 힘든가).
        - precipitating: 최근 촉발 요인/사건/시간 단서(무엇이 계기가 되었나).
        - perpetuating: 문제를 유지/악화시키는 반복 패턴·왜곡(자기비난/파국화/과잉일반화/당위문, '항상/계속' 등 반복 단서).
        - predisposing: 오래된 취약성/배경(과거부터 이어진 경향, 장기 맥락). 근거 없으면 출력하지 않음.
        - protective: 보호 요인/강점/지지(도움 요청, 가치·강점 언급, 지지 활용).
        - impact: 삶의 기능 영향(수면/학업·업무/대인 기능 저하 등 구체 영향).

        [작성 규칙]
        - 가능성/추정 표현(같다/듯/아마/가능성)은 쓰지 말 것.
        - '증거/스팬' 같은 메타 단어 금지.
        - 중복 스팬은 하나로 요약.

        [증거 스팬]
        {evidence_str}

        [출력 형식]
        - 정확히 한 문장만 출력. (빈 evidence면 공백 없이 빈 문자열만 출력)
        """


        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception:
        pass  # LLM 실패 시 fallback

    # fallback: 단순 evidence join
    return f"{slot}: {evidence_str}"


def generate_tom(features: Dict[str, Any], ppppi_cur: PPPPI) -> Dict[str, Any]:
    """
    ToM Generator:
      - affect_state: 로컬 수치
      - schemas: flag confidence
      - beliefs/desires: LLM 요약
      - intentions: 별도 LLM 호출 (BD + PPPPI 기반)
    """
    # affect_state 계산
    A_up = float(features.get("A_up", 0.0))
    A_down = float(features.get("A_down", 0.0))
    control = float(features.get("control", 0.0))
    dominance_low = round(1.0 - control, 6)
    affect_state = {
        "arousal_up": round(A_up, 6),
        "arousal_down": round(A_down, 6),
        "dominance_low": dominance_low,
    }

    # schemas 계산
    flags = features.get("flags", {}) or {}
    schemas = {}
    for name in ["self_blame", "catastrophizing", "overgeneralization", "should"]:
        info = flags.get(name) or {}
        if info.get("present"):
            schemas[name] = float(info.get("confidence", 1.0))
        else:
            schemas[name] = 0.0

    # PPPPI 텍스트
    def slot_text(slot: str) -> str:
        return (ppppi_cur.get(slot) or {}).get("text", "").strip()
    ctx_lines = [f"{s}: {slot_text(s)}" for s in SLOTS if slot_text(s)]
    context_ppppi = "\n".join(ctx_lines) if ctx_lines else "(empty)"

    # 1) beliefs/desires 생성
    bd = _generate_beliefs_desires(context_ppppi)

    # 2) intentions 생성 (beliefs, desires, PPPPI를 컨텍스트로)
    intentions = _generate_intentions(bd, context_ppppi)

    return {
        "beliefs": bd.get("beliefs", []),
        "desires": bd.get("desires", []),
        "intentions": intentions,
        "affect_state": affect_state,
        "schemas": schemas,
    }


def _generate_beliefs_desires(context_ppppi: str) -> Dict[str, List[str]]:
    """LLM으로 beliefs/desires만 생성"""
    model=set_gpt_model()
    client = OpenAI()
    sys = (
    "너는 PPPPI 정보에 기반하여 사용자의 belief와 desire를 해석하는 상담 ToM 요약기다. 반드시 아래 출력 규칙을 따른다.\n"
    "출력 규칙:\n"
    "1) 두 줄만 출력한다.\n"
    "2) 첫 줄: beliefs (내담자가 스스로 믿는 왜곡된 생각·해석, 최대 2개, 탭(\\t)으로 구분)\n"
    "   - 형식: 짧은 명사구 또는 '나는 ~' 구조 (예: '나는 항상 실패한다')\n"
    "   - 원문 전체 문장을 그대로 복사하지 말 것\n"
    "3) 둘째 줄: desires (내담자가 바라는 욕구·심리적 필요, 최대 2개, 탭(\\t)으로 구분)\n"
    "   - 형식: 짧은 욕구 문구 (예: '안정감이 필요하다', '죄책감에서 벗어나고 싶다')\n"
    "4) 따옴표, 라벨(beliefs:, desires:), JSON, 설명 문구는 금지.\n"
    "5) 출력은 문구만, 다른 아무것도 추가하지 말 것."
    )

    usr = (
    "설명:\n"
    "- beliefs → 내담자가 현재 상황을 해석하는 왜곡된 인지 패턴 (예: 자기비난, 과잉일반화)\n"
    "- desires → 내담자가 바라는 심리적 욕구/필요 (예: 안정감, 죄책감 해소, 관계 유지)\n\n"
    "[PPPPI 요약]\n"
    f"{context_ppppi}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": usr}],
        temperature=0.5,
    )
    content = (resp.choices[0].message.content or "").strip()

    lines = content.splitlines()
    beliefs = lines[0].split("\t") if len(lines) > 0 else []
    desires = lines[1].split("\t") if len(lines) > 1 else []

    # 공백 정리
    beliefs = [b.strip() for b in beliefs if b.strip()]
    desires = [d.strip() for d in desires if d.strip()]

    bd = {"beliefs": beliefs, "desires": desires}
    return bd


def _generate_intentions(bd: Dict[str, List[str]], context_ppppi: str) -> List[str]:
    """LLM으로 intentions만 생성 (탭 구분 포맷)"""
    model = set_gpt_model()
    client = OpenAI()

    sys = (
        "너는 상담 ToM 의도 추론기다. 단일 이론 틀인 Motivational Interviewing(MI)의 "
        "4가지 프로세스에 근거하여 다음 중에서만 선택한다:\n"
        "1) 공감·수용(Engaging): 관계형성/정서 공명/수용 강화\n"
        "2) 명료화(Focusing): 주제/목표/사실을 명확히 함\n"
        "3) 재구성(Evoking): 변화동기/새 관점 이끌어냄(가벼운 재구성)\n"
        "4) 행동계획(Planning): 구체적 다음 행동 합의\n\n"
        "출력 규칙:\n"
        "- 반드시 위 4개 중 최대 2개만 고른다. 1개만 해당될 시 1개만 고른다.\n"
        "- 출력은 오직 한 줄, 선택 항목을 탭(\\t)으로만 구분. 다른 텍스트/따옴표/마침표/JSON 금지.\n"
        "- 입력에 없는 정보는 추정/추가하지 않는다.\n"
        "- 불확실할 때는 안전 우선 순서(공감·수용 > 명료화 > 재구성 > 행동계획)를 따른다."
    )

    # --- User Prompt (의사결정 루브릭 + 입력) ---
    usr = (
        "다음 의사결정 루브릭을 적용하여 선택하라(최대 2개):\n"
        "- 공감·수용: 정서 각성이 높거나(불안/분노/죄책/슬픔 표현), 자기비난/파국화 등 고충 호소가 두드러질 때.\n"
        "- 명료화: 사건 경위/주체/목표/관계 맥락이 불명확하거나 PPPPI의 precipitating/impact가 빈약할 때.\n"
        "- 재구성: 정서가 어느 정도 안정되고(공감·수용 이후), 왜곡 신호가 1~2개 수준으로 관찰될 때 가볍게 시도.\n"
        "- 행동계획: 통제감 회복 의지/구체 목표 언급/실행 의향이 나타날 때.\n"
        "주의: 재구성과 행동계획은 공감·명료화 선행 후에 우선순위가 오른다.\n\n"
        "출력 형식(중요): 선택한 항목명만 탭(\\t)으로 구분하여 한 줄로 출력. 다른 문장/부연 금지.\n\n"
        f"beliefs: {bd.get('beliefs', [])}\n"
        f"desires: {bd.get('desires', [])}\n"
        f"PPPPI 요약:\n{context_ppppi}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": usr}],
        temperature=0.0,
    )

    content = (resp.choices[0].message.content or "").strip()
    # 탭 구분 파싱
    intentions = [c.strip() for c in content.split("\t") if c.strip()]

    # 안전 필터링 (허용된 집합만)
    allowed = {"공감·수용", "명료화", "재구성", "행동계획"}
    intentions = [it for it in intentions if it in allowed][:2]

    return intentions


