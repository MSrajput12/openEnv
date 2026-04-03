from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import json

from app.env import LogisticsEnv
from app.models import Action

app = FastAPI(title="LogisticsFlow OpenEnv")
env = LogisticsEnv()

# This tells the server it can accept a JSON body for the reset command
class ResetConfig(BaseModel):
    level: str = "easy"

# --- THE DASHBOARD ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    # We grab the current state to display on the dashboard
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
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📦 LogisticsFlow Environment <span class="badge">ONLINE</span></h1>
                <p>This is a headless OpenEnv API for AI Agent Evaluation. The environment simulates a dynamic supply chain.</p>
                <hr style="border-color: #30363d; margin: 20px 0;">
                <h3>📡 API Endpoints</h3>
                <ul>
                    <li><code>POST /reset</code> - Initialize a new episode (Pass <code>{{"level": "hard"}}</code> to test out-of-stock crises).</li>
                    <li><code>POST /step</code> - Execute an <code>Action</code> (ship, restock).</li>
                    <li><code>GET /state</code> - Retrieve the current environment state.</li>
                </ul>
                <h3>📊 Live World State:</h3>
                <pre><code>{state_json}</code></pre>
            </div>
        </body>
    </html>
    """

# --- THE OPENENV REQUIRED ENDPOINTS ---

# FIXED: Now accepts POST requests to pass the automated Validation Script
@app.post("/reset")
def reset_env_post(config: Optional[ResetConfig] = None):
    level = config.level if config else "easy"
    return env.reset(level)

# Kept the GET version just in case you want to test it in your browser URL bar
@app.get("/reset/{level}")
def reset_env_get(level: str):
    return env.reset(level)

@app.post("/step")
def step_env(action: Action):
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
def get_state():
    return env.state()

# --- THE OPENENV MULTI-MODE ENTRY POINT ---
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()