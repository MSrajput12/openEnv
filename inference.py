import os
import json
import urllib.request
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
TASK_NAME = "hard"
BENCHMARK = "LogisticsFlow-OpenEnv"

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
# 4. NETWORK HELPER (No 'requests' library needed)
# ==========================================
def send_post_request(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

# ==========================================
# 5. INFERENCE LOOP
# ==========================================
def run_inference():
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Reset Environment
    try:
        obs = send_post_request(f"{ENV_URL}/reset", {"level": TASK_NAME})
    except Exception as e:
        print(f"Failed to connect to environment: {e}")
        # The prompt instructed us to wrap risky operations. If it fails, we gracefully exit.
        return

    rewards = []
    steps_taken = 0
    success = False

    for step in range(1, 21): # Max 20 steps
        current_state = obs.get("observation", obs) if isinstance(obs, dict) else obs

        # Prepare prompt
        system_prompt = "You are an AI logistics agent. Analyze the state. You must output exactly valid JSON. Actions: {'command': 'ship', 'params': {'order_id': 'ORD-XYZ', 'carrier': 'Standard'}} OR {'command': 'restock', 'params': {'item': 'Electronics'}}."
        user_prompt = f"Current State: {json.dumps(current_state)}"

        # Get Model Action
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"} # Forces valid JSON
            )
            action_str = response.choices[0].message.content.strip()
            action_json = json.loads(action_str)
            action_log = action_str.replace('\n', '').replace(' ', '') # Flatten for logging
            error = None
        except Exception as e:
            action_json = {"command": "wait", "params": {}}
            action_log = "error"
            error = str(e)

        # Step the Environment
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
        
        # Log strictly
        log_step(step=step, action=action_log, reward=reward, done=done, error=error)

        if done:
            break

    # Calculate final score (clamp between 0 and 1 for judges)
    score = sum(rewards)
    normalized_score = min(max(score, 0.0), 1.0)
    success = normalized_score > 0.0

    log_end(success=success, steps=steps_taken, score=normalized_score, rewards=rewards)

if __name__ == "__main__":
    run_inference()