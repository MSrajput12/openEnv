from app.models import State
from typing import Dict, Any, Tuple

class LogisticsTasks:
    
    @staticmethod
    def get_task_setup(level: str) -> Tuple[Dict, list, float]:
        """Returns (inventory, pending_orders, budget) for the chosen difficulty."""
        if level == "easy":
            return (
                {"Electronics": 10}, 
                [{"id": "ORD-1", "priority": "Standard", "age": 0}], 
                100.0
            )
        elif level == "medium":
            return (
                {"Apparel": 20}, 
                [{"id": "ORD-1", "priority": "VIP", "age": 0}, 
                 {"id": "ORD-2", "priority": "Standard", "age": 0}], 
                15.0  # Tight budget! AI must use cheap shipping.
            )
        elif level == "hard":
            return (
                {"Electronics": 0},  # OUT OF STOCK! AI must restock first.
                [{"id": "ORD-1", "priority": "VIP", "age": 0}], 
                100.0
            )
        else:
            return ({"Default": 10}, [], 100.0)

class TaskGrader:
    
    @staticmethod
    def grade(level: str, initial_state: State, final_state: State) -> float:
        """Scores the AI's performance from 0.0 to 1.0"""
        score = 0.0
        
        # Did they clear all orders? (Base requirement)
        orders_cleared = len(initial_state.pending_orders) - len(final_state.pending_orders)
        total_orders = len(initial_state.pending_orders)
        
        if total_orders > 0:
            score += (orders_cleared / total_orders) * 0.5  # 50% of score is just finishing the job
            
        # Task Specific Grading
        if level == "easy":
            # Just finish the job
            if len(final_state.pending_orders) == 0:
                score += 0.5
                
        elif level == "medium":
            # Must finish job AND not go bankrupt
            if len(final_state.pending_orders) == 0 and final_state.budget >= 0:
                score += 0.5
            elif final_state.budget < 0:
                score = 0.0 # Punish bankruptcy
                
        elif level == "hard":
            # Must restock and ship VIP
            if len(final_state.pending_orders) == 0:
                score += 0.5
            if final_state.inventory.get("Electronics", 0) > 0:
                # Bonus points if they restocked properly and have leftover
                score += 0.1
                
        # Cap score between 0.0 and 1.0
        return max(0.0, min(1.0, score))