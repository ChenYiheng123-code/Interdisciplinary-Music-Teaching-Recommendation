
import json
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def read_prompt(name):
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def fallback_report(path_names, diag_notes, corrections):
    steps = "; ".join(f"step {i + 1}: {n}" for i, n in enumerate(path_names))
    return dict(report=f"Planned path -> {steps}. Diagnostics: {' '.join(diag_notes) or 'n/a'} {' '.join(corrections)}".strip(),
                fallback=True)


class DummyLLM:
    """Deterministic stand-in used by tests / smoke runs (no weights)."""
    name = "dummy"

    def generate(self, prompt, max_new_tokens=512):
        return json.dumps({"report": "DUMMY report generated from the planned path.", "fallback": False})


class QwenLLM:
    def __init__(self, name="Qwen/Qwen2.5-7B-Instruct", revision=None, dtype="float16", max_new_tokens=512, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.max_new_tokens = name, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=getattr(torch, dtype)).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)                                   # frozen
        self.device = device

    def generate(self, prompt, max_new_tokens=None):
        import torch
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def explain_once(llm, path_names, node_diag, phase_plan, corrections, fb_names):
    """Single LLM call. Returns (report_dict, used_fallback)."""
    prompt = read_prompt("explainer.txt").format(
        path="\n".join(f"{i + 1}. {n}" for i, n in enumerate(path_names)),
        diagnostics=json.dumps(node_diag, ensure_ascii=False), phase_plan=phase_plan,
        corrections="\n".join(corrections) or "none")
    raw = llm.generate(prompt)
    js = parse_json(raw)
    if js is None or "report" not in js:
        return fallback_report(path_names, fb_names, corrections), True
    return js, False
