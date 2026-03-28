from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Action(BaseModel):
    command: str = Field(..., description="Action: 'ship', 'restock', or 'wait'")
    params: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    inventory: Dict[str, int]
    pending_orders: List[Dict[str, Any]]
    budget: float
    message: str

class State(BaseModel):
    inventory: Dict[str, int]
    pending_orders: List[Dict[str, Any]]
    budget: float
    steps_taken: int
    max_steps: int