import json
from typing import Dict, List, Any, Optional

from openai import OpenAI
import openai
from ..utils import set_gpt_model, set_openai_api_key


set_openai_api_key()

def safe_json_parse(text: str) -> dict:
    """LLM 출력에서 마지막 JSON 객체만 안전 파싱."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}\s*$", text, re.S)
        if not m:
            raise ValueError(f"Model did not return JSON:\n{text[:300]}...")
        return json.loads(m.group(0))



def get_llm_response(system_prompt,prompt):
    client = OpenAI()
    model = set_gpt_model()
    response = client.chat.completions.create(
        model=model,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content": system_prompt},
                  {"role":"user","content": prompt}]
    )
    return safe_json_parse(response.choices[0].message.content)


def create_empty_memory() -> Dict[str, Any]:
    """새로운 빈 메모리 구조 생성"""
    return {
        "turn_history": [],
            "overall_summary": {'summary':{
                "core_narrative": "",
                "core_emotion": [],
                "recurring_themes": []},
            "critical_events": [],
            "ppppi_synthesis": {
                "presenting": "",
                "precipitating": "",
                "perpetuating": "",
                "predisposing": "",
                "protective": "",
                "impact": ""
            }
        }
    }



# ====== 1. Turn Analysis Prompts ======
def create_turn_analysis_prompts(user_input, tom) -> tuple[str, str]:
    """Turn Analysis용 system_prompt와 prompt 생성"""
    
    system_prompt =  """당신은 상담 대화 분석 전문가입니다. 사용자의 발화와 분석된 신호 정보를 종합하여 상담에 필요한 정보를 추출하세요.

분석 기준:
- Summary: 발화의 핵심 내용을 상담적 관점에서 요약
- Keywords: 감정, 사건, 중요 개념, 인지 패턴 위주로 추출
- Events: 구체적으로 언급된 사건들만 포함
- Emotions: 신호 분석과 텍스트를 종합한 감정 상태

현재 턴 신호와 누적 패턴을 모두 고려하여 종합적으로 분석하세요.
반드시 JSON 형식으로만 응답하세요."""
    
    prompt = f"""사용자 발화: "{user_input}"

    Congitive Flag : {tom['tom_state']['schemas']}

다음 JSON 형식으로 응답해주세요:
{{
    "summary": "이 턴의 핵심 내용을 1-2문장으로 요약",
    "keywords": ["주요", "키워드", "리스트"],
    "events": [
        {{
            "event": "구체적 사건 설명",
            "context": "사건의 맥락",
            "impact_level": "high|medium|low"
        }}
    ],
    "emotions": [
        {{
            "emotion": "감정명",
            "trigger": "감정의 트리거"
        }}
    ]
}}"""
    
    return system_prompt, prompt

# ====== 2. Critical Events Update Prompts ======
def create_critical_events_prompts(current_events: List[Dict], new_events: List[Dict], new_emotions: List[Dict]) -> tuple[str, str]:
    """Critical Events 업데이트용 prompts 생성"""
    
    system_prompt = """당신은 상담 기록 관리 전문가입니다. 기존 중요 사건들과 새로운 사건들을 비교하여 체계적으로 업데이트하세요.

업데이트 원칙:
1. 기존 사건과 유사한 새 사건이 있으면 emotions 배열에 추가
2. 완전히 새로운 사건이면 새 항목으로 추가
3. 감정은 반드시 각 사건과 인과적으로 관련있는 감정을 채택
4. 감정은 시간 순서대로 배열 (과거 → 현재)
5. 중복을 피하고 의미있는 변화만 기록

반드시 JSON 형식으로만 응답하세요."""
    
    prompt = f"""기존 Critical Events:
{current_events}

새로운 Events:
{new_events}

새로운 감정 :
{new_emotions}

JSON 형식으로 응답:
[
    {{
        "event": "이벤트 설명",
        "emotions": ["시간순으로", "나열된", "감정들"]
    }}
]"""
    
    return system_prompt, prompt

# ====== 3. PPPPI Summary Update Prompts ======
# ====== 3. PPPPI Summary Update Prompts (revised) ======
from typing import Dict, Tuple

def create_ppppi_update_prompts(
    current_ppppi: Dict[str, dict],
    turn_analysis: Dict,
    current_tom: Dict
) -> Tuple[str, str]:
    """
    PPPPI 업데이트용 prompts 생성 (추측 마킹 및 증거 병기 포함)

    입력 예시
    - current_ppppi: {
        "presenting": {"text":"...", "evidence":["..."], "is_inferred":0, "confidence":0.8, "changed":0, "provenance":["history"]},
        ...
      }
    - turn_analysis: 최신 턴에서 추출된 사실/스팬/맥락
    - current_tom: 최신 ToM (beliefs/desires/intentions/affect 등)

    반환: (system_prompt, user_prompt)
    """

    system_prompt = """당신은 상담 사례 공식화 전문가입니다. PPPPI 모델을 사용해 내담자 정보를 '증거 기반'으로 점진 업데이트하세요.

원칙(매우 중요):
1) '증거 기반' 업데이트만: 새로운 증거(직접 발화 스팬·사실)가 없으면 기존 내용을 유지합니다.
2) 추측 마킹: 직접적 증거 없이 합리적 추정으로 채울 때는 is_inferred=1, confidence<=0.5로 표기합니다.
3) 증거 병기: 가능한 한 evidence에는 '짧은 스팬/키워드'를 담습니다(원문 요약 스팬). 없으면 빈 배열.
4) 최소 수정: 바뀐 슬롯만 changed=1, 나머지는 changed=0으로 반환합니다.
5) 보수적 언어: 진단·범주화·라벨링은 금지. 해결책/방법 나열 유도 금지.
6) 시간·맥락·행동 연결이 분명할 때만 구체화하세요(없으면 추측으로 채우지 말고 유지).
7) 단정 금지: 확률적·가설적일 경우 is_inferred=1로 표시하고, 과도한 확신을 피하세요.

출력은 오직 하나의 JSON만. JSON 외 텍스트 금지.
슬롯 키: presenting, precipitating, perpetuating, predisposing, protective, impact.

슬롯 정의와 예시(복사 금지):
- Presenting: 현재 내담자가 호소하는 문제 
  (예: "잠을 잘 못 잔다", "불안이 심하다")
- Precipitating: 최근 발생한 촉발 요인 
  (예: "시험 직전", "친구와의 갈등 후")
- Perpetuating: 문제를 유지/악화시키는 반복적 패턴 
  (예: "완벽주의로 계속 압박을 받음", "수면 부족의 악순환")
- Predisposing: 오래된 성향이나 배경 요인 
  (예: "어릴 때부터 불안 경향", "가족력")
- Protective: 보호 요인이나 자원 
  (예: "친구의 지지", "규칙적 운동")
- Impact: 일상 기능에 미치는 영향 
  (예: "집중이 안 돼 학업 성적 저하", "대인관계 회피")

각 슬롯 객체 스키마:
{
  "text": "핵심 내용(한두 문장, 과장 금지)",
  "evidence": ["근거 스팬1","근거 스팬2"],     // 없으면 []
  "is_inferred": 0 or 1,                       // 증거 없거나 간접 추정이면 1
  "confidence": 0.0~1.0,                       // 주관적 확신도; 추측이면 <=0.5
  "changed": 0 or 1,                           // 이번 턴에서 변경/보강되었는가
  "provenance": ["turn_analysis","tom","history"] 중 포함 // 출처 표시
}"""

    user_prompt = f"""현재 PPPPI(이전 상태, 그대로 참고·재사용 가능):
{json.dumps(current_ppppi, ensure_ascii=False, indent=2)}

새로운 TOM 상태(최신):
{json.dumps(current_tom, ensure_ascii=False, indent=2)}

새로운 Turn Analysis(최신 턴에서 얻은 추가 정보·스팬·사실):
{json.dumps(turn_analysis, ensure_ascii=False, indent=2)}

요청:
- 새로운 증거가 있는 슬롯만 내용을 보강/수정하고 changed=1로 표시하세요.
- 새로운 증거가 없으면 해당 슬롯을 이전 값 그대로 유지하고 changed=0으로 두세요.
- 직접 근거 없이 채우는 경우에는 is_inferred=1, confidence<=0.5로 표시하세요.
- evidence는 '짧은 스팬' 위주로 정리.
- 진단/낙인/치료조언/방법 나열 유도 금지.

반드시 아래 JSON 스키마로만 응답:
{{
  "presenting": {{"text":"", "evidence":[], "is_inferred":0, "confidence":0.0, "changed":0, "provenance":[]}},
  "precipitating": {{"text":"", "evidence":[], "is_inferred":0, "confidence":0.0, "changed":0, "provenance":[]}},
  "perpetuating": {{"text":"", "evidence":[], "is_inferred":0, "confidence":0.0, "changed":0, "provenance":[]}},
  "predisposing": {{"text":"", "evidence":[], "is_inferred":0, "confidence":0.0, "changed":0, "provenance":[]}},
  "protective": {{"text":"", "evidence":[], "is_inferred":0, "confidence":0.0, "changed":0, "provenance":[]}},
  "impact": {{"text":"", "evidence":[], "is_inferred":0, "confidence":0.0, "changed":0, "provenance":[]}}
}}"""

    return system_prompt, user_prompt


# ====== 4. Overall Narrative Update Prompts ======
def create_overall_narrative_prompts(current_summary: Dict, turn_analysis: Dict, updated_events: List, updated_ppppi: Dict) -> tuple[str, str]:
    """Overall Narrative 업데이트용 prompts 생성"""
    
    system_prompt = """당신은 상담 전체 흐름을 종합하는 전문가입니다. 개별 정보들을 통합하여 내담자의 전체적인 상황을 파악하세요.

종합 원칙:
- Core Narrative: 내담자의 전체 상황을 간결하고 명확하게 요약
- Core Emotion: 가장 지배적이고 지속적인 감정들 (3-5개)
- Recurring Themes: 반복적으로 나타나는 주요 주제들

기존 정보와 새 정보를 자연스럽게 통합하여 일관성 있는 전체상을 만드세요.
반드시 JSON 형식으로만 응답하세요."""
    
    prompt = f"""현재 Overall Summary:
- Core Narrative: {current_summary.get('core_narrative', '')}
- Core Emotion: {current_summary.get('core_emotion', [])}
- Recurring Themes: {current_summary.get('recurring_themes', [])}

새로운 Turn Analysis:
{turn_analysis}

업데이트된 Critical Events:
{updated_events}

업데이트된 PPPPI:
{updated_ppppi}

JSON 형식으로 응답:
{{
    "core_narrative": "이전 narrative와 새 정보를 자연스럽게 통합한 1-2문장",
    "core_emotion": ["가장", "지배적인", "감정들"],
    "recurring_themes": ["반복되는", "주요", "테마들"]
}}"""
    
    return system_prompt, prompt


def build_memory(user_input, tom, memory):
    system_prompt_turn, prompt_turn = create_turn_analysis_prompts(user_input, tom)
    turn_information = get_llm_response(system_prompt_turn, prompt_turn)
    
    memory['turn_history'].append(turn_information)

    
    current_events = memory['overall_summary']['critical_events']
    new_events = turn_information['events']
    new_emotions = turn_information['emotions']

    system_prompt_ce, prompt_ce = create_critical_events_prompts(current_events, new_events, new_emotions)
    critical_events = get_llm_response(system_prompt_ce, prompt_ce)
    
    memory['overall_summary']['critical_events'] = critical_events


    current_ppppi = memory['overall_summary']["ppppi_synthesis"]
    current_tom = tom['ppppi']

    system_prompt_ppppi, prompt_ppppi = create_ppppi_update_prompts(current_ppppi, turn_information, current_tom)
    ppppi_synthesis = get_llm_response(system_prompt_ppppi, prompt_ppppi)

    memory['overall_summary']["ppppi_synthesis"] = ppppi_synthesis


    current_summary = memory['overall_summary']['summary']
    system_prompt_os, prompt_os  = create_overall_narrative_prompts(current_summary, turn_information, critical_events, ppppi_synthesis)
    overall_summary = get_llm_response(system_prompt_os, prompt_os )
    memory['overall_summary']['summary'] = overall_summary 

    return memory



    