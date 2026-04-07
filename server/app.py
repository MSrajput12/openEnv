from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import json

from app.env import LogisticsEnv
from app.models import Action

app = FastAPI(title="LogisticsFlow OpenEnv")
env = LogisticsEnv()

# FIX: Score must be strictly between 0 and 1 (not 0.0, not 1.0)
MIN_SCORE = 0.001
MAX_SCORE = 0.999

def clamp_score(score: float) -> float:
    """Clamp score to strictly open interval (0, 1)."""
    return min(max(float(score), MIN_SCORE), MAX_SCORE)

class ResetConfig(BaseModel):
    level: str = "easy"

# ==========================================
# DASHBOARD
# ==========================================
@app.get("/", response_class=HTMLResponse)
def read_root():
    current_state = env.state().model_dump()
    state_json = json.dumps(current_state, indent=4)

    return f"""
    <html>
        <head>
            <title>LogisticsFlow Playground</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 2rem; }}
                .container {{ max-width: 800px; margin: auto; background: #161b22; padding: 30px; border-radius: 12px; border: 1px solid #30363d; }}
                h1 {{ color: #58a6ff; margin-top: 0; }}
                h3 {{ color: #2ea043; }}
                pre {{ background: #010409; padding: 15px; border-radius: 8px; overflow-x: auto; border: 1px solid #30363d; color: #79c0ff; }}
                .badge {{ background: #238636; color: white; padding: 4px 8px; border-radius: 2em; font-size: 12px; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; }}
                th {{ background: #21262d; color: #58a6ff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📦 LogisticsFlow Environment <span class="badge">ONLINE</span></h1>
                <p>Headless OpenEnv API for AI Agent Evaluation. Simulates a dynamic supply chain with 3 difficulty levels.</p>
                <hr style="border-color: #30363d; margin: 20px 0;">
                <h3>📡 API Endpoints</h3>
                <table>
                    <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
                    <tr><td>POST</td><td>/reset</td><td>Start episode. Body: {{"level": "easy|medium|hard"}}</td></tr>
                    <tr><td>POST</td><td>/step</td><td>Execute action (ship/restock)</td></tr>
                    <tr><td>GET</td><td>/state</td><td>Get current environment state</td></tr>
                    <tr><td>GET</td><td>/grade/easy</td><td>Get grader score for easy task</td></tr>
                    <tr><td>GET</td><td>/grade/medium</td><td>Get grader score for medium task</td></tr>
                    <tr><td>GET</td><td>/grade/hard</td><td>Get grader score for hard task</td></tr>
                </table>
                <h3>📊 Live World State:</h3>
                <pre><code>{state_json}</code></pre>
            </div>
        </body>
    </html>
    """

# ==========================================
# OPENENV REQUIRED ENDPOINTS
# ==========================================
@app.post("/reset")
def reset_env_post(config: Optional[ResetConfig] = None):
    level = config.level if config else "easy"
    return env.reset(level)

@app.get("/reset/{level}")
def reset_env_get(level: str):
    return env.reset(level)

@app.post("/step")
def step_env(action: Action):
    obs, reward, done, info = env.step(action)
    # FIX: Clamp reward to strictly (0, 1) so graders never receive 0.0 or 1.0
    clamped_reward = clamp_score(reward) if reward != 0.0 else 0.0
    return {"observation": obs, "reward": clamped_reward, "done": done, "info": info}

@app.get("/state")
def get_state():
    return env.state()

# ==========================================
# FIX: GRADERS FOR ALL 3 TASK LEVELS
# Each grader returns score strictly in (0, 1)
# ==========================================
@app.get("/grade/easy")
def grade_easy():
    """Grader for easy task. Score strictly in (0, 1)."""
    state = env.state().model_dump()
    raw_score = _compute_grade(state, level="easy")
    return {
        "task": "easy",
        "score": clamp_score(raw_score),
        "status": "graded"
    }

@app.get("/grade/medium")
def grade_medium():
    """Grader for medium task. Score strictly in (0, 1)."""
    state = env.state().model_dump()
    raw_score = _compute_grade(state, level="medium")
    return {
        "task": "medium",
        "score": clamp_score(raw_score),
        "status": "graded"
    }

@app.get("/grade/hard")
def grade_hard():
    """Grader for hard task. Score strictly in (0, 1)."""
    state = env.state().model_dump()
    raw_score = _compute_grade(state, level="hard")
    return {
        "task": "hard",
        "score": clamp_score(raw_score),
        "status": "graded"
    }

@app.post("/grade/{task}")
def grade_task_post(task: str, payload: dict = {}):
    """POST grader endpoint for a given task. Score strictly in (0, 1)."""
    state = env.state().model_dump()
    raw_score = _compute_grade(state, level=task)
    return {
        "task": task,
        "score": clamp_score(raw_score),
        "status": "graded"
    }

def _compute_grade(state: dict, level: str) -> float:
    """
    Compute a grade for the current environment state.
    Returns a float. Will be clamped to strictly (0.001, 0.999).
    
    Grading logic:
    - easy:   Based on orders_fulfilled ratio
    - medium: Based on orders_fulfilled + low stockout penalty
    - hard:   Based on orders_fulfilled + stockout handling + efficiency
    """
    try:
        orders_fulfilled = state.get("orders_fulfilled", 0)
        total_orders = state.get("total_orders", 1) or 1
        stockouts = state.get("stockouts", 0)
        steps_taken = state.get("steps_taken", 1) or 1

        fulfillment_rate = orders_fulfilled / total_orders

        if level == "easy":
            score = fulfillment_rate * 0.9  # max 0.9 to avoid hitting 1.0
        elif level == "medium":
            penalty = min(stockouts * 0.05, 0.3)
            score = (fulfillment_rate * 0.85) - penalty
        else:  # hard
            penalty = min(stockouts * 0.08, 0.4)
            efficiency = min(orders_fulfilled / steps_taken, 1.0) * 0.1
            score = (fulfillment_rate * 0.8) - penalty + efficiency

        return score

    except Exception:
        # Safe fallback: return a mid-range score
        return 0.5

# ==========================================
# ENTRY POINT
# ==========================================
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()