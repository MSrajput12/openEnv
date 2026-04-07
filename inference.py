import os
import json
import urllib.request
import time
from openai import OpenAI

# ==========================================
# 1. MANDATORY ENVIRONMENT VARIABLES
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
HF_TOKEN = os.getenv("HF_TOKEN")

# ==========================================
# 2. ENVIRONMENT SETUP
# ==========================================
ENV_URL = "https://mark012-logisticsflow-openenv.hf.space"
BENCHMARK = "LogisticsFlow-OpenEnv"

# FIX 1: Define all 3 required tasks with graders
TASKS = ["easy", "medium", "hard"]

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ==========================================
# 3. STRICT LOGGING FORMATTERS
# ==========================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ==========================================
# 4. NETWORK HELPER (No 'requests' library)
# ==========================================
def send_post_request(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

# ==========================================
# 5. FIX 2: Score must be STRICTLY between 0 and 1
# ==========================================
def compute_score(rewards: list) -> float:
    raw = sum(rewards)
    # Clamp to strictly (0.0, 1.0) — 0.0 and 1.0 are NOT allowed
    MIN_SCORE = 0.001
    MAX_SCORE = 0.999
    return min(max(raw, MIN_SCORE), MAX_SCORE)

# ==========================================
# 6. SINGLE TASK RUNNER
# ==========================================
def run_task(task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Reset Environment for this task level
    try:
        obs = send_post_request(f"{ENV_URL}/reset", {"level": task_name})
    except Exception as e:
        print(f"Failed to connect to environment for task '{task_name}': {e}", flush=True)
        log_end(success=False, steps=0, score=0.001, rewards=[])
        return 0.001

    rewards = []
    steps_taken = 0

    for step in range(1, 21):  # Max 20 steps per task
        current_state = obs.get("observation", obs) if isinstance(obs, dict) else obs

        system_prompt = (
            "You are an AI logistics agent. Analyze the state carefully. "
            "You must output exactly valid JSON. "
            "Available actions: "
            "{'command': 'ship', 'params': {'order_id': 'ORD-XYZ', 'carrier': 'Standard'}} "
            "OR {'command': 'restock', 'params': {'item': 'Electronics'}}."
        )
        user_prompt = f"Current State: {json.dumps(current_state)}"

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            action_str = response.choices[0].message.content.strip()
            action_json = json.loads(action_str)
            action_log = action_str.replace('\n', '').replace(' ', '')
            error = None
        except Exception as e:
            action_json = {"command": "wait", "params": {}}
            action_log = "error"
            error = str(e)

        try:
            obs = send_post_request(f"{ENV_URL}/step", action_json)
            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
        except Exception as e:
            reward = 0.0
            done = True
            error = str(e)

        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action_log, reward=reward, done=done, error=error)

        # Rate limit safety
        time.sleep(4)

        if done:
            break

    # FIX 2: Use strict (0, 1) scorer
    score = compute_score(rewards)
    success = score > 0.001

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

# ==========================================
# 7. MAIN: Run ALL 3 tasks
# ==========================================
def run_inference():
    all_scores = {}

    for task in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"Running task: {task}", flush=True)
        print(f"{'='*50}", flush=True)
        score = run_task(task)
        all_scores[task] = score
        # Brief pause between tasks to avoid rate limits
        time.sleep(5)

    # Summary
    print(f"\n[SUMMARY] All task scores:", flush=True)
    for task, score in all_scores.items():
        print(f"  {task}: {score:.3f}", flush=True)

if __name__ == "__main__":
    run_inference()