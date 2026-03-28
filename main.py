from fastapi import FastAPI
from app.env import LogisticsEnv
from app.models import Action
from app.tasks import TaskGrader

app = FastAPI()
env = LogisticsEnv()

# Store initial state for grading
initial_state_cache = {}

@app.get("/reset/{level}")
def reset(level: str):
    obs = env.reset(level)
    initial_state_cache["last"] = env.state()
    return obs

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/grade")
def grade():
    if "last" not in initial_state_cache:
        return {"error": "Must reset environment first"}
    
    score = TaskGrader.grade(env.current_level, initial_state_cache["last"], env.state())
    return {"level": env.current_level, "final_score": score}