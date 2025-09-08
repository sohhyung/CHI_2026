import pandas as pd
from openai import OpenAI
import openai
from ..utils import set_gpt_model, set_openai_api_key
import json
import re

set_openai_api_key()

kmi_df = pd.read_csv("utils/models/custom_utils/data/kmi.csv")

def get_examples(query_text: str, k=3):
    """
    query_text: 최근 사용자 발화
    k: 라벨별로 뽑을 예시 개수
    """
    examples = {}
    for lab, grp in kmi_df.groupby("label"):
        # 단순 키워드 겹침으로 스코어링
        q_tokens = set(query_text.split())
        grp = grp.assign(
            score=grp["user_utterance"].apply(lambda t: len(q_tokens & set(t.split())))
        )
        topk = grp.sort_values("score", ascending=False).head(k)
        examples[lab] = list(zip(topk["user_utterance"], topk["therapist_utterance"]))

    return examples


def build_prompt(user_text, context, examples, allowed_labels, exclude_labels=None):
    exclude_labels = set(exclude_labels or [])
    ex_blocks = []
    for lab, exs in examples.items():
        if lab in exclude_labels:
            continue
        if not exs:
            continue
        # (user, therapist) 페어를 출력한다고 가정
        lines = "\n".join(
            f"- [USER≈] {u}\n  [THERAPIST→{lab}] {t}"
            for (u, t) in exs
        )
        ex_blocks.append(f"[{lab} EXAMPLES]\n{lines}")
    ex_str = "\n\n".join(ex_blocks)

    visible_labels = [lab for lab in allowed_labels if lab not in exclude_labels]

    return f"""
[RECENT USER TEXT]
{user_text}

[CONTEXT]
{context}

[ALLOWED LABELS]
{visible_labels}

[FEW-SHOT EXAMPLES]
{ex_str}

Return JSON only:
{{"label":"<one_label>","rationale":"<why next agent utterance should be in this label in 1-2 Korean sentences>"}}
"""




def safe_json_parse(text: str) -> dict:
    """Extract JSON object from model output string."""
    try:
        return json.loads(text)
    except:
        # 중괄호 블록만 추출
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise

REFLECTION_GROUP = {"Simple Reflection", "Complex Reflection"}
QUESTION_GROUP   = {"Open Question", "Closed Question"}

def expanded_excludes(primary_label: str) -> list[str]:
    if primary_label in REFLECTION_GROUP:
        return list(REFLECTION_GROUP)
    if primary_label in QUESTION_GROUP:
        return list(QUESTION_GROUP)
    # 그 외(Advise / Give Information / Affirm / General)는 자기 자신만 제외
    return [primary_label]



def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def predict_labels(user_text, context):
    model=set_gpt_model()
    client=OpenAI()

    allowed = kmi_df["label"].unique().tolist()
    exs = get_examples(user_text, k=2)  # (user, therapist) 튜플로 뽑히도록 구현돼 있어야 함

    # 1차
    prompt1 = build_prompt(user_text, context, exs, allowed, exclude_labels=None)
    res1 = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=200,
        messages=[
            {"role": "system", "content": "You are an MI label selector. Return ONLY JSON."},
            {"role": "user", "content": prompt1}
        ]
    )
    primary = safe_json_parse(res1.choices[0].message.content)
    primary_label = primary.get("label")

    # 2차 — 리플렉션/질문 짝 라벨까지 함께 제외
    excludes = expanded_excludes(primary_label)
    prompt2 = build_prompt(user_text, context, exs, allowed, exclude_labels=excludes)
    res2 = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=200,
        messages=[
            {"role": "system", "content": "You are an MI label selector. Return ONLY JSON."},
            {"role": "user", "content": prompt2}
        ]
    )
    secondary = safe_json_parse(res2.choices[0].message.content)

    result = {
        "label": primary_label,
        "secondary_label": secondary.get("label"),
        "rationale": {
            "primary": primary.get("rationale",""),
            "secondary": secondary.get("rationale","")
        }
    }

    return exs, result
