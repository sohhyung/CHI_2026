from typing import List, Dict, Any, Optional
import os, re, json, random
from ..utils import set_gpt_model, set_openai_api_key

from openai import OpenAI

set_openai_api_key()


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def to01(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    return clip01((v - lo) / float(hi - lo))

def mean_ignore_none(vals):
    xs = [x for x in vals if x is not None]
    return sum(xs)/len(xs) if xs else 0.0


def build_baseline_signals(intake: Dict[str, Any]) -> Dict[str, Any]:
    PANAS_KEYS = {
    "anxiety": ["불안하다", "긴장된다", "초조하다"],
    "anger":   ["짜증난다", "짜증스럽다", "화가 난다"],
    "sadness": ["우울하다", "슬프다"],
    "guilt":   ["죄책감을 느낀다"],
    "fatigue": ["지친다"],}

    p = intake.get("panas_na", {}) or {}

    def n(key: str):
        v = p.get(key)
        return clip01(float(v)/5.0) if v is not None else None

    affect_baseline = {sub: mean_ignore_none([n(k) for k in keys]) for sub, keys in PANAS_KEYS.items()}

    control_baseline    = to01(intake.get("control"),    0, 3) or 0.0
    discomfort_baseline = to01(intake.get("discomfort"), 0, 100) or 0.0
    importance_baseline = to01(intake.get("importance"), 1, 6) or 0.0

    cat = intake.get("category","") or ""
    domain =cat
    return {
        "affect_baseline": affect_baseline,
        "control_baseline": control_baseline,
        "discomfort_baseline": discomfort_baseline,
        "importance_baseline": importance_baseline,
        "domain": domain
    }



AFFECT_KEYS = ["anxiety_t", "anger_t", "sadness_t", "hurt_t", "embarrass_t"]
FLAGS = ["catastrophizing","overgeneralization","self_blame","should"]

def affect_predict_proba(text: str) -> List[float]:
    AFFECT_LEX = {
        "anxiety_t": ["불안","걱정","긴장","초조","두렵"],
        "anger_t": ["화","짜증","분노","억울"],
        "sadness_t": ["슬프","우울","무기력","눈물","허무"],
        "hurt_t": ["상처","소외","배신","모멸","무시"],
        "embarrass_t": ["당황","민망","부끄","혼란"],
    }
    # Replace with your calibrated BERT 5-class classifier
    scores = []
    for k in AFFECT_KEYS:
        hits = sum(1 for w in AFFECT_LEX[k] if re.search(w, text))
        val = clip01(0.1 + 0.25*hits + 0.05*random.random())  # demo-only heuristic
        scores.append(val)
    s = sum(scores) or 1.0
    return [clip01(x/s) for x in scores]

def affect_text_scores(text: str) -> Dict[str, float]:
    return dict(zip(AFFECT_KEYS, affect_predict_proba(text)))


def regex_hits(text: str) -> Dict[str, List[Dict[str, Any]]]:
    PREFILTER_PATTERNS = {
    "catastrophizing": r"(망했|끝났|큰일|최악|재앙|폭망|인생이\s*끝)",
    "overgeneralization": r"(항상|늘|절대|전부|아무도|매번|영원히)",
    "self_blame": r"(내\s*탓|내\s*잘못|내가\s*문제|내가\s*망쳤|내가\s*실수)",
    "should": r"(반드시|꼭|필히|해야만|해야\s*한다|절대.*안\s*된다)",
    }
    hits = {f: [] for f in FLAGS}
    for k, pat in PREFILTER_PATTERNS.items():
        for m in re.finditer(pat, text):
            hits[k].append({"text": m.group(0), "start": m.start(), "end": m.end()})
    return hits

def llm_call_json(topic_text: str, hint_spans: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:

    model = set_gpt_model()
    temperature = 0.2
    client = OpenAI()


    FLAGS = ["catastrophizing","overgeneralization","self_blame","should"]

    # 공통 프롬프트
    sys_prompt = (
        "너는 상담 발화에서 인지 왜곡을 판정하는 구조화 추출기다. hint를 참고하여 아래의 정의에 해당하는 내용이 있다면 추출하여 양식에 맞추어 출력한다."
        "정의: catastrophizing=재난/최악 단정, "
        "overgeneralization=항상/절대/전부/아무도/매번 등 일반화, "
        "self_blame=내 탓/내 잘못/내가 문제/내가 망쳤/내가 실수, "
        "should=반드시/꼭/필히/~해야만/~해야 한다/절대 ~하면 안 된다. "
        "농담/인용/타인 발언은 제외하고, 모호하면 present=false로 하라."
    )
    user_payload = {
        "topic_text": topic_text,
        "hint_spans": hint_spans  # 프리필터 후보 스팬(없어도 OK)
    }

    # 1) Responses API + json_schema 시도
    json_schema = {
        "name": "cognitive_flags",
        "schema": {
            "type": "object",
            "properties": {
                "flags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":    {"type": "string", "enum": FLAGS},
                            "present": {"type": "boolean"},
                            "spans": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text":  {"type": "string"},
                                        "start": {"type": "integer"},
                                        "end":   {"type": "integer"}
                                    },
                                    "required": ["text","start","end"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["name","present","spans"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["flags"],
            "additionalProperties": False
        },
        "strict": True
    }

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role":"system","content":sys_prompt},
                {"role":"user","content":json.dumps(user_payload, ensure_ascii=False)}
            ],
            response_format={"type":"json_schema","json_schema":json_schema,"strict":True},
            temperature=temperature,
        )
        if getattr(resp, "output_parsed", None):
            return resp.output_parsed
        txt = getattr(resp, "output_text", "") or ""
        return json.loads(txt) if txt else {"flags": []}

    except TypeError:
        # 2) Chat Completions + Function Calling (SDK 1.x 전반 호환)
        tools = [{
            "type": "function",
            "function": {
                "name": "set_cognitive_flags",
                "description": "Extract cognitive distortion flags from counseling utterance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flags": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name":    {"type": "string", "enum": FLAGS},
                                    "present": {"type": "boolean"},
                                    "spans": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "text":  {"type": "string"},
                                                "start": {"type": "integer"},
                                                "end":   {"type": "integer"}
                                            },
                                            "required": ["text","start","end"],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                "required": ["name","present","spans"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["flags"],
                    "additionalProperties": False
                }
            }
        }]

        cc = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role":"system","content": sys_prompt + " 반드시 함수 호출로만 답하라."},
                {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            tools=tools,
            tool_choice={"type":"function","function":{"name":"set_cognitive_flags"}}
        )

        # 함수 호출 인자(JSON) 파싱
        call = cc.choices[0].message.tool_calls[0]
        args = call.function.arguments  # str(JSON)
        return json.loads(args)




def aggregate_votes(outs: List[Dict[str, Any]]):
    votes = {f: 0 for f in FLAGS}
    span_lists = {f: [] for f in FLAGS}
    for o in outs:
        for it in o.get("flags", []):
            if it.get("present"): votes[it["name"]] += 1
            span_lists[it["name"]].extend(it.get("spans", []))
    return votes, span_lists

def bayes_conf(S: int, k: int, alpha: float=1.0, beta: float=1.0) -> float:
    return clip01((S + alpha) / (k + alpha + beta))

def iou_char(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = max(0, min(a["end"],b["end"]) - max(a["start"],b["start"]))
    union = max(a["end"],b["end"]) - min(a["start"],b["start"])
    return inter/union if union>0 else 0.0

def span_consistency(spans: List[Dict[str, Any]]) -> float:
    if len(spans) < 2: return 1.0 if spans else 0.0
    ious = []
    for i in range(len(spans)):
        for j in range(i+1, len(spans)):
            ious.append(iou_char(spans[i], spans[j]))
    return sum(ious)/len(ious) if ious else 1.0

def pick_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uniq = {(s.get("text",""), s.get("start",0), s.get("end",0)): s for s in spans}
    cands = list(uniq.values())
    cands.sort(key=lambda s: (s.get("end",0) - s.get("start",0)))
    return cands[:2]

def cognitive_flags(text: str,
                    k_base: int = 3,
                    k_max: int = 5,
                    upgrade_conf_range = (0.4, 0.6),
                    length_upgrade_threshold: int = 200,
                    lambda_span: float = 0.8) -> Dict[str, Dict[str, Any]]:
    hints = regex_hits(text)
    outs = [llm_call_json(text, hints) for _ in range(k_base)]
    votes3, spans3 = aggregate_votes(outs)

    conf3 = {}
    for f in FLAGS:
        alpha,beta = (2.0,1.0) if hints[f] else (1.0,1.0)
        base = bayes_conf(votes3[f], k_base, alpha, beta)
        sc = span_consistency(spans3[f])
        conf3[f] = clip01(lambda_span*base + (1-lambda_span)*sc)

    need_upgrade = any((votes3[f] < 2) or (upgrade_conf_range[0] <= conf3[f] <= upgrade_conf_range[1]) for f in FLAGS) \
                   or (len(text) > length_upgrade_threshold)

    if need_upgrade and k_max > k_base:
        outs += [llm_call_json(text, hints) for _ in range(k_max - k_base)]

    n = len(outs)
    votesN, spansN = aggregate_votes(outs)

    decisions = {}
    for f in FLAGS:
        S = votesN[f]
        alpha,beta = (2.0,1.0) if hints[f] else (1.0,1.0)
        base = bayes_conf(S, n, alpha, beta)
        sc = span_consistency(spansN[f])
        confidence = clip01(lambda_span*base + (1-lambda_span)*sc)
        decisions[f] = {
            "present": S >= (n//2 + 1),
            "confidence": round(confidence, 3),
            "uncertainty": round(1-confidence, 3),
            "spans": pick_spans(spansN[f])
        }
    return decisions

# ============================
# TurnSignals builder
# ============================

def build_turn_signals(topic_text: str) -> Dict[str, Any]:
    return {
        "affect_text": affect_text_scores(topic_text),
        "cognitive_flags": cognitive_flags(topic_text)
    }

