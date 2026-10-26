"""Pilot Diagnostic v2 - Uses currently available Groq model"""
import json, sys, os, time, traceback, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from caps.config import get_groq_api_keys
from caps.llm.groq_client import GroqProvider
from caps.datasets.synthetic_v2 import generate_synthetic_v2_dataset
from caps.environments.synthetic_v2 import SyntheticV2Environment
from caps.agents.receiver import ReceiverOutput
from caps.communication.ciu import CIU

OUT = Path(__file__).parent / "diagnostic_trace_v2.json"
print("=" * 60)
print("PILOT DIAGNOSTIC v2")
print("=" * 60)

keys = get_groq_api_keys()
print(f"\n[1] Keys: {len(keys)}")

# Use the best available model
MODEL = "openai/gpt-oss-120b"
provider = GroqProvider(api_key=keys, default_model=MODEL)
print(f"    Model: {MODEL}")

print("\n[2] API test...")
try:
    test_resp = provider.generate(prompt="What is 2+2? Reply with just the number.", system_prompt="Be brief.", max_tokens=10, temperature=0.0)
    print(f"    OK: '{test_resp.text}' | p={test_resp.token_usage.prompt_tokens} c={test_resp.token_usage.completion_tokens} | {test_resp.latency_ms:.0f}ms")
    api_works = True
except Exception as e:
    print(f"    FAIL: {e}")
    api_works = False
    sys.exit(1)

print("\n[3] Dataset...")
tasks = generate_synthetic_v2_dataset(size=10, seed=42)
easy = [t for t in tasks if t.get("difficulty") == "easy"] or tasks
task = easy[0]

print(f"    Task: {task['task_id']}")
print(f"    answer_key: {task['answer_key']}")
print(f"    answer_text: {task['answer_text']}")

env = SyntheticV2Environment(); env.load_tasks([task]); env.reset(0)
cius = env.get_cius()
print(f"    CIUs: {len(cius)}")

context_parts = [f"--- [ID: {c.id}] [TYPE: {c.type}] ---\n{c.content}" for c in cius]
context_str = "\n\n".join(context_parts)
receiver_prompt = f"TASK:\n{env.get_task_description()}\n\nPROVIDED CONTEXT:\n{context_str}\n\nInstructions:\n1. Solve the task using ONLY the provided context.\n2. If critical information is missing, indicate what is needed.\n3. Respond with a JSON object:\n{{\n  \"status\": \"success\" or \"failure\" or \"needs_clarification\",\n  \"answer\": \"your solution/answer\",\n  \"used_ciu_ids\": [\"list of CIU IDs you actually used\"],\n  \"missing_information\": [\"list of what you need but do not have\"],\n  \"confidence\": 0.0 to 1.0\n}}"
system_prompt = f"You are a {task.get('receiver_role', 'general')} agent. Solve the task using ONLY the provided context. Do not hallucinate. Always respond with valid JSON."

print(f"\n[4] Calling LLM ({MODEL})...")
t0 = time.time()
response = provider.generate(prompt=receiver_prompt, system_prompt=system_prompt, model=MODEL, temperature=0.0, max_tokens=800, response_schema=ReceiverOutput)
elapsed = (time.time() - t0) * 1000
print(f"    Latency: {elapsed:.0f}ms | Error: {response.error}")
print(f"    Tokens: p={response.token_usage.prompt_tokens} c={response.token_usage.completion_tokens}")
print(f"\n    RAW:\n    {response.text[:1500]}")

print(f"\n[5] Parse & Evaluate...")
try:
    parsed = json.loads(response.text)
    answer_text = parsed.get("answer", response.text)
    print(f"    status: {parsed.get('status')}")
    print(f"    answer: {answer_text[:300]}")
    print(f"    used_ciu_ids: {parsed.get('used_ciu_ids', [])}")
except json.JSONDecodeError:
    answer_text = response.text
    print(f"    JSON parse failed, using raw text")

success = env.evaluate(answer_text)
print(f"    answer_key='{task['answer_key']}' | MATCH={success}")
if not success:
    key_lower = task['answer_key'].lower()
    ans_lower = answer_text.lower()
    print(f"    Manual: '{key_lower}' in answer? {key_lower in ans_lower}")
    print(f"    Regex word boundary? {bool(re.search(r'\\b' + re.escape(key_lower) + r'\\b', ans_lower))}")

# Now test 3 more tasks to get a quick sense
print(f"\n[6] Running 3 more tasks...")
results = [{"task": task["task_id"], "success": success, "model": MODEL}]
for t in tasks[1:4]:
    env2 = SyntheticV2Environment(); env2.load_tasks([t]); env2.reset(0)
    cius2 = env2.get_cius()
    parts2 = [f"--- [ID: {c.id}] [TYPE: {c.type}] ---\n{c.content}" for c in cius2]
    ctx2 = "\n\n".join(parts2)
    p2 = f"TASK:\n{env2.get_task_description()}\n\nPROVIDED CONTEXT:\n{ctx2}\n\nInstructions:\n1. Solve the task using ONLY the provided context.\n2. If critical information is missing, indicate what is needed.\n3. Respond with a JSON object:\n{{\n  \"status\": \"success\" or \"failure\" or \"needs_clarification\",\n  \"answer\": \"your solution/answer\",\n  \"used_ciu_ids\": [\"list of CIU IDs you actually used\"],\n  \"missing_information\": [\"list of what you need but do not have\"],\n  \"confidence\": 0.0 to 1.0\n}}"
    s2 = f"You are a {t.get('receiver_role', 'general')} agent. Solve the task using ONLY the provided context. Do not hallucinate. Always respond with valid JSON."
    try:
        r2 = provider.generate(prompt=p2, system_prompt=s2, model=MODEL, temperature=0.0, max_tokens=800, response_schema=ReceiverOutput)
        try:
            p2d = json.loads(r2.text)
            a2 = p2d.get("answer", r2.text)
        except:
            a2 = r2.text
        ok2 = env2.evaluate(a2)
        print(f"    {t['task_id'][:50]}: success={ok2} | key={t['answer_key']} | answer[:80]={a2[:80]}")
        results.append({"task": t["task_id"], "success": ok2, "model": MODEL})
    except Exception as e:
        print(f"    {t['task_id'][:50]}: ERROR: {e}")
        results.append({"task": t["task_id"], "success": False, "error": str(e), "model": MODEL})

trace = {"api_works": api_works, "model": MODEL, "results": results}
OUT.write_text(json.dumps(trace, indent=2), encoding="utf-8")
print(f"\n[7] Trace saved: {OUT}")
n_ok = sum(1 for r in results if r.get("success"))
print(f"\n    SUMMARY: {n_ok}/{len(results)} tasks succeeded")
print("=" * 60)
