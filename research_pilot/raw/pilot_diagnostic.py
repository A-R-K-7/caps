"""
Pilot Diagnostic Script - Single-Task Real-LLM Trace
"""
import json, sys, os, time, traceback, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from caps.config import get_groq_api_keys
from caps.llm.groq_client import GroqProvider
from caps.datasets.synthetic_v2 import generate_synthetic_v2_dataset
from caps.environments.synthetic_v2 import SyntheticV2Environment
from caps.agents.receiver import ReceiverAgent, ReceiverOutput
from caps.communication.ciu import CIU

OUT = Path(__file__).parent / "diagnostic_trace.json"
print("=" * 60)
print("PILOT DIAGNOSTIC - SINGLE TASK REAL-LLM TRACE")
print("=" * 60)

keys = get_groq_api_keys()
print(f"\n[1] API keys found: {len(keys)}")
if not keys:
    print("ERROR: No API keys found."); sys.exit(1)

provider = GroqProvider(api_key=keys, default_model="llama-3.3-70b-versatile")
print(f"    Provider: GroqProvider, model: {provider.default_model}")

print("\n[2] Validating API connection...")
try:
    test_resp = provider.generate(prompt="What is 2+2? Reply with just the number.", system_prompt="Be brief.", max_tokens=10, temperature=0.0)
    print(f"    Response: '{test_resp.text}', tokens: p={test_resp.token_usage.prompt_tokens} c={test_resp.token_usage.completion_tokens}, latency: {test_resp.latency_ms:.0f}ms, error: {test_resp.error}")
    api_works = True
except Exception as e:
    print(f"    ERROR: {e}"); traceback.print_exc(); api_works = False

print("\n[3] Generating dataset (size=10)...")
tasks = generate_synthetic_v2_dataset(size=10, seed=42)
easy = [t for t in tasks if t.get("difficulty") == "easy"] or tasks
task = easy[0]

print(f"\n[4] Task: {task['task_id']}")
print(f"    Category: {task['category']}, Difficulty: {task['difficulty']}")
print(f"    Answer key: {task['answer_key']}")
print(f"    Answer text: {task['answer_text']}")
gt = task.get("ground_truth", {})
print(f"    Necessary: {gt.get('necessary_cius', [])}")

env = SyntheticV2Environment(); env.load_tasks([task]); env.reset(0)
cius = env.get_cius()
print(f"\n[5] CIUs: {len(cius)}")
for c in cius:
    print(f"      {c.id}: type={c.type}, tokens={c.estimate_tokens()}")

context_parts = [f"--- [ID: {c.id}] [TYPE: {c.type}] ---\n{c.content}" for c in cius]
context_str = "\n\n".join(context_parts)
receiver_prompt = f"TASK:\n{env.get_task_description()}\n\nPROVIDED CONTEXT:\n{context_str}\n\nInstructions:\n1. Solve the task using ONLY the provided context.\n2. If critical information is missing, indicate what is needed.\n3. Respond with a JSON object:\n{{\n  \"status\": \"success\" or \"failure\" or \"needs_clarification\",\n  \"answer\": \"your solution/answer\",\n  \"used_ciu_ids\": [\"list of CIU IDs you actually used\"],\n  \"missing_information\": [\"list of what you need but do not have\"],\n  \"confidence\": 0.0 to 1.0\n}}"
system_prompt = f"You are a {task.get('receiver_role', 'general')} agent. Solve the task using ONLY the provided context. Do not hallucinate information that was not provided. Always respond with valid JSON."

print(f"\n[6] Prompt: {len(receiver_prompt)} chars")

print(f"\n[7] Calling LLM...")
t0 = time.time()
try:
    response = provider.generate(prompt=receiver_prompt, system_prompt=system_prompt, model=provider.default_model, temperature=0.0, max_tokens=800, response_schema=ReceiverOutput)
    elapsed = (time.time() - t0) * 1000
    print(f"    Latency: {elapsed:.0f}ms, Error: {response.error}")
    print(f"    Tokens: p={response.token_usage.prompt_tokens} c={response.token_usage.completion_tokens}")
    print(f"\n    RAW RESPONSE:\n    {response.text[:1500]}")
except Exception as e:
    print(f"    ERROR: {e}"); traceback.print_exc(); response = None; elapsed = 0

if response:
    print(f"\n[8] Parsing...")
    try:
        parsed = json.loads(response.text)
        answer_text = parsed.get("answer", response.text)
        print(f"    status: {parsed.get('status')}, answer: {answer_text[:200]}")
        print(f"    used_ciu_ids: {parsed.get('used_ciu_ids', [])}")
    except json.JSONDecodeError:
        print(f"    JSON parse failed, using raw text"); answer_text = response.text
    
    print(f"\n[9] Evaluating...")
    success = env.evaluate(answer_text)
    print(f"    answer_key='{task['answer_key']}' | match={success}")
    if not success:
        key_lower = task['answer_key'].lower()
        ans_lower = answer_text.lower()
        print(f"    Manual: key in answer? {key_lower in ans_lower}")
        print(f"    Regex: {bool(re.search(r'\\b' + re.escape(key_lower) + r'\\b', ans_lower))}")
    
    trace = {"api_works": api_works, "model": provider.default_model, "task_id": task["task_id"], "answer_key": task["answer_key"], "answer_text": task["answer_text"], "raw_response": response.text, "error": response.error, "latency_ms": elapsed, "tokens": {"prompt": response.token_usage.prompt_tokens, "completion": response.token_usage.completion_tokens}, "success": success}
    OUT.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"\n[10] Trace saved: {OUT}")

print("\n" + "=" * 60)
