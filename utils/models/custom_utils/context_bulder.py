from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json

KMI_LABELS = [
    "Simple Reflection","Complex Reflection","Open Question","Closed Question",
    "Affirm","Give Information","Advise","General"
]

# ----------------------------
# Helpers
# ----------------------------
def _get(d: Dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _round(x, n=2):
    try:
        return round(float(x), n)
    except Exception:
        return x

# ----------------------------
# Core builders
# ----------------------------
def detect_missing_slots(tom: Dict[str, Any]) -> List[str]:
    """Check PPPPI slots that are missing."""
    slots = ["presenting","precipitating","perpetuating","predisposing","protective","impact"]
    missing = []
    ppppi = tom.get("ppppi", {}) or {}
    for s in slots:
        t = _get(ppppi, [s, "text"], "")
        if not isinstance(t, str) or t.strip() == "":
            missing.append(s)
    return missing

def extract_affect(turn_signal: Dict[str, Any]) -> Dict[str, float]:
    aff = turn_signal.get("affect_text", {}) or {}
    out = {}
    for k, v in aff.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out

def extract_flags(turn_signal: Dict[str, Any]) -> Dict[str, Any]:
    flags = turn_signal.get("cognitive_flags", {}) or {}
    flat = {}
    span_texts = []
    for name, info in flags.items():
        present = bool(info.get("present", False))
        conf = info.get("confidence")
        flat[f"{name}"] = present
        if conf is not None:
            try:
                flat[f"{name}_confidence"] = float(conf)
            except Exception:
                flat[f"{name}_confidence"] = conf
        for sp in info.get("spans", []) or []:
            t = sp.get("text", "")
            if t and isinstance(t, str):
                span_texts.append(t.strip())
    flat["_salient_spans"] = span_texts
    return flat

def summarize_tom(tom: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "beliefs": tom.get("tom_state", {}).get("beliefs"),
        "desires": tom.get("tom_state", {}).get("desires"),
        "intentions": tom.get("tom_state", {}).get("intentions"),
        "schemas": tom.get("tom_state", {}).get("schemas"),
        "affect_state": tom.get("tom_state", {}).get("affect_state"),
        "ppppi_presenting": _get(tom, ["ppppi","presenting","text"], ""),
        "ppppi_missing": detect_missing_slots(tom),
    }

def build_context(
    basic_signal: Dict[str, Any],
    turn_signals: List[Dict[str, Any]],
    tom0: Dict[str, Any],
    tom1: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = {
        "domain": basic_signal.get("basic_signal", {}).get("domain") or basic_signal.get("domain"),
        "control_baseline": basic_signal.get("basic_signal", {}).get("control_baseline", basic_signal.get("control_baseline", 0.0)),
        "discomfort_baseline": basic_signal.get("basic_signal", {}).get("discomfort_baseline", basic_signal.get("discomfort_baseline", 0.0)),
        "importance_baseline": basic_signal.get("basic_signal", {}).get("importance_baseline", basic_signal.get("importance_baseline", 0.0)),
        "affect_baseline": basic_signal.get("basic_signal", {}).get("affect_baseline", basic_signal.get("affect_baseline", {})),
    }
    last_ts = turn_signals[-1] if turn_signals else {}
    aff = extract_affect(last_ts)
    flags = extract_flags(last_ts)

    t0 = summarize_tom(tom0)
    t1 = summarize_tom(tom1)

    ctx = {
        "domain": baseline["domain"],
        "baseline": {
            "control": _round(baseline["control_baseline"]),
            "discomfort": _round(baseline["discomfort_baseline"]),
            "importance": _round(baseline["importance_baseline"]),
            "affect_baseline": {k: _round(v) for k, v in (baseline["affect_baseline"] or {}).items()},
        },
        "tom0": t0,
        "tom1": t1,
        "turnsignals": {
            "affect_text": {k: _round(v) for k, v in aff.items()},
            "cognitive_flags": {k: v for k, v in flags.items() if not k.startswith("_")},
            "salient_spans": flags.get("_salient_spans", []),
        },
        "missing_slots": t1["ppppi_missing"],
    }
    return ctx

# ----------------------------
# Prompt-format helpers
# ----------------------------
def format_context_block(ctx: Dict[str, Any]) -> str:
    bl = ctx["baseline"]
    t0, t1 = ctx["tom0"], ctx["tom1"]
    ts = ctx["turnsignals"]
    missing = ctx.get("missing_slots", [])

    def _fmt(d): 
        return json.dumps(d, ensure_ascii=False)

    lines = [
        f"Domain: {ctx.get('domain','')}",
        f"Baseline: control={bl['control']}, discomfort={bl['discomfort']}, importance={bl['importance']}",
        f"ToM(0): desires={_fmt(t0.get('desires'))}, intentions={_fmt(t0.get('intentions'))}, schemas={_fmt(t0.get('schemas'))}",
        f"ToM(-1): presenting=\"{t1.get('ppppi_presenting','')}\", desires={_fmt(t1.get('desires'))}, intentions={_fmt(t1.get('intentions'))}, schemas={_fmt(t1.get('schemas'))}",
        f"TurnSignals(-1): affect={_fmt(ts.get('affect_text'))}, flags={_fmt(ts.get('cognitive_flags'))}",
    ]
    return "\n".join(lines)

def build_query_text(recent_user_text: str, ctx: Dict[str, Any], max_len: int = 800) -> str:
    spans = ctx["turnsignals"].get("salient_spans", [])
    span = f" {spans[0]}" if spans else ""
    presenting = ctx["tom1"].get("ppppi_presenting", "")
    q = f"{recent_user_text}{span} {presenting}".strip()
    return q[:max_len]

# ----------------------------
# Test harness
# ----------------------------
def demo_build_context(
    basic_signal: Dict[str, Any],
    turn_signal_list: List[Dict[str, Any]],
    tom_list: List[Dict[str, Any]],
    recent_user_text: str
) -> Tuple[Dict[str, Any], str, str]:
    tom0 = tom_list[0] if tom_list else {}
    tom1 = tom_list[-1] if tom_list else {}
    ctx = build_context(basic_signal, turn_signal_list, tom0, tom1)
    block = format_context_block(ctx)
    query = build_query_text(recent_user_text, ctx)
    return ctx, block, query
