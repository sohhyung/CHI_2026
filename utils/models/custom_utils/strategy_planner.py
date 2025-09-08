from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import json, re

from openai import OpenAI
import openai
from ..utils import set_gpt_model, set_openai_api_key

set_openai_api_key()
# -------------------- 공통/유틸 --------------------

def safe_json_parse(text: str) -> dict:
    """LLM 출력에서 마지막 JSON 객체만 안전 파싱."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}\s*$", text, re.S)
        if not m:
            raise ValueError(f"Model did not return JSON:\n{text[:300]}...")
        return json.loads(m.group(0))

def format_examples_block(
    examples: Dict[str, List[Tuple[str, str]]],
    labels_to_include: List[str],
    per_label_k: int = 3) -> str:
    """라벨별 (user, therapist) 페어를 프롬프트용 블록으로 포맷."""
    blocks=[]
    for lab in labels_to_include:
        pairs = examples.get(lab, []) or []
        if not pairs:
            continue
        lines=[]
        for (u, t) in pairs[:per_label_k]:
            lines.append(f"- [USER≈] {u}\n  [THERAPIST→{lab}] {t}")
        blocks.append(f"[{lab} EXAMPLES]\n" + "\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no examples)"

# -------------------- OpenQ rationale 추출 --------------------
def build_strategy_context_block(tom, memory) -> str:

    t0, tN = tom[0], tom[-1]

    # --- ToM: turn0 / last ---
    b0 = t0.get("tom_state", {}).get("beliefs", [])
    d0 = t0.get("tom_state", {}).get("desires", [])
    i0 = t0.get("tom_state", {}).get("intentions", [])
    bN = tN.get("tom_state", {}).get("beliefs", [])
    dN = tN.get("tom_state", {}).get("desires", [])
    iN = tN.get("tom_state", {}).get("intentions", [])

    # --- Memory parts ---
    last_sum, last_events, last_emotions = "-", [], []
    core_narr, core_emotions, themes, ppppi_syn = "-", [], [], {}

    if isinstance(memory, dict):
        th = memory.get("turn_history") or []
        if th:
            last = th[-1]
            last_sum = (last.get("summary") or "-").strip()

            # events 요약 (event + impact_level)
            for e in last.get("events", []):
                event = e.get("event")
                level = e.get("impact_level")
                if event:
                    if level:
                        last_events.append(f"{event} (impact level: {level})")
                    else:
                        last_events.append(event)

            # emotions 요약 (emotion + trigger)
            for em in last.get("emotions", []):
                emo = em.get("emotion")
                trig = em.get("trigger")
                if emo:
                    if trig:
                        last_emotions.append(f"{emo} (trigger: {trig})")
                    else:
                        last_emotions.append(emo)

        osum = memory.get("overall_summary") or {}
        sblock = osum.get("summary") or {}
        core_narr = (sblock.get("core_narrative") or "-").strip()
        core_emotions = sblock.get("core_emotion") or []
        themes = sblock.get("recurring_themes") or []
        ppppi_syn = osum.get("ppppi_synthesis") or {}

    def _join(xs): 
        return ", ".join([str(x) for x in xs if str(x).strip()]) or "-"

    # PPPPI synthesis formatting
    ppppi_lines = []
    for slot in ["presenting","precipitating","perpetuating","predisposing","protective","impact"]:
        block = ppppi_syn.get(slot, {}) or {}
        txt = block.get("text", "").strip()
        evid = block.get("evidence", []) or []
        if txt:
            ppppi_lines.append(f"- {slot}: {txt} (evidence {len(evid)}개)")
    ppppi_block = "\n".join(ppppi_lines) if ppppi_lines else "(no PPPPI synthesis)"

    return (
        "[CONTEXT — ToM & Memory 요약]\n"
        "아래는 사용자의 상태(ToM: beliefs/desires/intentions)와 메모리 요약입니다.\n\n"
        "[ToM — INITIAL (turn 0)]\n"
        f"- beliefs: {_join(b0)}\n"
        f"- desires: {_join(d0)}\n"
        f"- intentions: {_join(i0)}\n\n"
        "[ToM — RECENT (last turn)]\n"
        f"- beliefs: {_join(bN)}\n"
        f"- desires: {_join(dN)}\n"
        f"- intentions: {_join(iN)}\n\n"
        "[MEMORY — Last Turn Summary]\n"
        f"{last_sum}\n"
        f"- events: {_join(last_events)}\n"
        f"- emotions: {_join(last_emotions)}\n\n"
        "[MEMORY — Overall Summary]\n"
        f"- core narrative: {core_narr}\n"
        f"- core emotions: {_join(core_emotions)}\n"
        f"- recurring themes: {_join(themes)}\n\n"
        "[MEMORY — PPPPI synthesis]\n"
        f"{ppppi_block}"
    )


# ============================================================
# (4) Slot Selector  — Open Question일 때만 호출 (question_type 제거)
# ============================================================

DEFAULT_QUESTIONS = {
    "impact": [
        "이 경험이 일상에서 어떤 부분에 가장 영향을 주고 있나요?",
        "요즘 생활 리듬이나 관계에 어떤 변화가 느껴지시나요?"
    ],
    "precipitating": [
        "최근에 이렇게 불편감이 커지게 된 계기나 상황이 있었을까요?",
        "언제부터 특히 어려움이 커졌다고 느끼셨나요?"
    ],
    "perpetuating": [
        "이 상태가 지속되거나 심해지는 패턴이 있다면 어떤 때인가요?",
        "특정 상황에서 특히 더 힘들어지는 경향이 있나요?"
    ],
    "protective": [
         "그 와중에도 잠깐 숨 고를 수 있었던 순간이 있었다면, 무엇이 그렇게 만들어줬나요?",
           "요즘 버팀목이 되어준 사람·장소·작은 루틴이 있었다면 떠오르는 게 있나요?"
    ]
}


# ============================================================
# (5) Strategy Planner  — 문장 없이 설계도(blueprint) JSON 생성
# ============================================================
STRATEGY_SYSTEM = """당신은 MI(동기 면담) 전략 계획자입니다.
오직 하나의 유효한 JSON만 반환하세요 (마지막 문장 없이). 한국어 라벨/노트 허용.
MI 정신(공감, 자율성, 협력)을 따르세요.

금지 규칙:
- Open Question에서 내담자에게 '방법/해결책을 나열하게 하는' 질문 금지
  (예: "어떤 방법이 도움이 될까요?", "무슨 방법이 있을까요?" 등 금지)
- Open Question은 방법 탐색형(“어떤 방법/무슨 방법…”)으로 작성하지 말고, 영향/패턴/의미/요인/상황에 대한 탐색형으로 작성하세요.
- 허락 확인형 표현(괜찮으실까요 등) 금지

어떤 행동이 정보 제공 또는 조언인 경우:
permission_required=true로 설정
permission_mode를 "embed" 또는 "invitational" 중 하나로 설정
연속적으로 여러 질문을 쌓는 것을 피하세요. 간결하게 유지하세요.
"""

def build_strategy(
    recent_user_text: str,
    context_block: Optional[str],
    label_result: Dict[str, Any],
    examples_block: str,
) -> str:
    primary = label_result.get("label")
    secondary = label_result.get("secondary_label")
    r_p = (label_result.get("rationale",{}) or {}).get("primary","")
    r_s = (label_result.get("rationale",{}) or {}).get("secondary","")
    ctx = (context_block or "(no additional context)").strip()


    # focus taxonomy: 재진술 계열 포함
    schema = """{
        "plan": {
            "speech_acts": ["<primary>", "<optional_secondary>"],
            "tone": "warm-empathetic",
            "target_slot": null | "impact" | "precipitating" | "perpetuating" | "protective",
            "moves": [
            {"act":"<primary>","goal":"<why this act>"},
            {"act":"<optional_secondary>","goal":"<why this act>"}
            ],
            "reason": "<short reason>",
            "guardrails": {"no_advice_without_permission": true, "avoid_stack_questions": true}
        },
        "act_plans": [
            {
            "act":"<label>",
            "focus":[
                "restatement_basic","restatement_expanded",
                "emotion_reflection","meaning_expand",
                "open_probe","fact_check",
                "self_efficacy","info","advice","bridge"
            ],
            "key_points":["<bullet 1>","<bullet 2>"],
            "style_hints":["<hint1>","<hint2>"],
            "permission_required": true|false,
            "permission_mode": "embed|invitational",                    
            "constraints":["1-2 sentences","supportive","non-judgmental","no diagnosis"]
            }
        ],
        "rg_contract": {
            "advice_permission_mode": "embed|invitational",               
            "macros":{
            "EMBED": ["도움이 되신다면","가능하시다면","여건이 된다면"],  
            "INVITATIONAL": ["…도 한 가지 방법이에요","…해보실 수도 있겠어요","…도 한 가지 선택지예요"]  
            },
            "length_total":"1-2 sentences",
            "banned":["진단적 표현","과도한 확신"]
        }
        }"""

    return f"""
[RECENT USER TEXT]
{recent_user_text}

[CONTEXT]
사용자의 ToM(믿음·욕구·의도)와 메모리 요약 정보입니다.
{ctx}

[LABEL RESULT]
primary="{primary}", secondary="{secondary}"
rationale.primary="{r_p}"
rationale.secondary="{r_s}"


[STYLE HINTS (FEW-SHOT FROM SELECTED LABELS)]
{examples_block}

[OUTPUT RULES]
- 주어진 primary/secondary를 plan.speech_acts로 사용하세요 (secondary가 존재하고 구별되는 경우).
- Open Question이 포함된 경우: ppppi 중에서 질문을 할 plan.target_slot을 설정하고, 해당 act_plan에 "open_probe" 포커스를 포함하세요.
- 마지막 문장을 생성하지 마세요; 청사진 필드만 작성하세요.
- 행동이 "Give Information" 또는 "Advise"인 경우: permission_required=true로 설정하세요.
- 모든 것을 간결하게 유지하고; 예시 문장을 그대로 복사하지 마세요.

[OUTPUT JSON SCHEMA]
{schema}
"""

def call_strategy_blueprint(prompt) -> dict:
    client=OpenAI()
    model=set_gpt_model()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content": STRATEGY_SYSTEM},
                  {"role":"user","content": prompt}]
    )
    return safe_json_parse(resp.choices[0].message.content)



ADVICE_ACTS = {"Give Information","Advise"}

def choose_permission_mode_2_3(ctx: dict, user_text: str, act: str) -> str:
    """
    embed vs invitational 중 선택.
    기본: invitational
    - 불편감↑/통제감↓ : embed
    - 사용자 측 명시 요청(조언/방법/추천 등 키워드) : invitational 유지
    """
    if act not in ADVICE_ACTS:
        return "invitational"

    mode = "invitational"
    try:
        discomfort = (ctx.get("baseline", {}) or {}).get("discomfort", 0.0)
        dominance_low = (ctx.get("tom1", {}).get("affect_state", {}) or {}).get("dominance_low", 0.0)
    except Exception:
        discomfort, dominance_low = 0.0, 0.0

    if discomfort >= 0.85 or dominance_low >= 0.3:
        mode = "embed"

    if any(k in (user_text or "") for k in ["방법 알려", "조언", "어떻게 해야", "알려줘", "추천해"]):
        mode = "invitational"

    return mode


# -------------------- Orchestrator --------------------
def run_strategy_planner(
    recent_user_text: str,
    label_result: Dict[str, Any],
    examples: Dict[str, List[Tuple[str, str]]],
    context_block: Optional[str] = None,
    ctx:dict |None=None
) -> Dict[str, Any]:
    """
    1) (조건) Open Question이면 Slot Selector 호출
    2) Strategy Blueprint 생성
    반환: {"blueprint": <dict>, "slot_selection": <dict|None>}
    """
    selected_labels = [l for l in [label_result.get("label"), label_result.get("secondary_label")] if l]
    examples_block = format_examples_block(examples, selected_labels, per_label_k=2)


    # (B) Strategy Planner (Blueprint 생성)
    strat_prompt = build_strategy(
        recent_user_text=recent_user_text,
        context_block=context_block,
        label_result=label_result,
        examples_block=examples_block,
    )
    blueprint = call_strategy_blueprint(strat_prompt)

    # (C) 최소 보정
    plan = blueprint.get("plan", {}) or {}
    acts = plan.get("speech_acts", []) or []
    uniq = []
    for a in acts:
        if a and a not in uniq:
            uniq.append(a)
    plan["speech_acts"] = uniq


    # act_plans 정렬 및 permission_required 보정
    order = {a:i for i,a in enumerate(uniq)}
    aps = blueprint.get("act_plans", []) or []
    aps.sort(key=lambda x: order.get(x.get("act",""), 99))
    for ap in aps:
        act = ap.get("act","")
        if act in ADVICE_ACTS:
            ap["permission_required"] = True
            ap["permission_mode"] = choose_permission_mode_2_3(
                ctx=ctx if isinstance(ctx, dict) else {},   # ← 호출부에서 전달 가능한 ctx 사용
                user_text=recent_user_text,
                act=act
            )
    blueprint["act_plans"] = aps

    rgc = blueprint.get("rg_contract", {}) or {}
    rgc["advice_permission_mode"] = rgc.get("advice_permission_mode", "invitational")
    blueprint["rg_contract"] = rgc

    return {"blueprint": blueprint}
