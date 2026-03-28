from app.models import Action, Observation, State

class LogisticsEnv:
    def __init__(self):
        self.max_steps = 20
        self.current_level = "easy"
        self.reset()

    def reset(self, level: str = "easy") -> Observation:
        from app.tasks import LogisticsTasks
        
        inv, orders, budget = LogisticsTasks.get_task_setup(level)
        self.inventory = inv
        self.pending_orders = orders
        self.budget = budget
        self.steps_taken = 0
        self.current_level = level
        
        return self._get_obs(f"Warehouse Reset to {level.upper()} mode.")

    def _get_obs(self, msg: str) -> Observation:
        return Observation(
            inventory=self.inventory,
            pending_orders=self.pending_orders,
            budget=self.budget,
            message=msg
        )

    def state(self) -> State:
        return State(
            inventory=self.inventory,
            pending_orders=self.pending_orders,
            budget=self.budget,
            steps_taken=self.steps_taken,
            max_steps=self.max_steps
        )

    def step(self, action: Action):
        self.steps_taken += 1
        reward = 0.0
        msg = "Action processed."
        
        # --- LOGIC FOR SHIPPING ---
        if action.command == "ship":
            order_id = action.params.get("order_id")
            order = next((o for o in self.pending_orders if o["id"] == order_id), None)
            
            if order:
                # Check if we have inventory to ship!
                item_needed = order.get("item", "Electronics") # Default to Electronics if not specified
                if self.inventory.get(item_needed, 0) > 0:
                    self.inventory[item_needed] -= 1
                    self.pending_orders.remove(order)
                    reward = 1.0 if order["priority"] == "VIP" else 0.5
                    msg = f"Shipped {order_id}. Remaining inventory: {self.inventory[item_needed]}"
                else:
                    reward = -0.2
                    msg = f"Cannot ship {order_id}. Out of stock of {item_needed}!"
            else:
                reward = -0.1
                msg = "Order not found"
                
        # --- LOGIC FOR RESTOCKING (This was missing!) ---
        elif action.command == "restock":
            item = action.params.get("item", "Electronics")
            cost = 20.0 # It costs money to restock
            
            if self.budget >= cost:
                self.budget -= cost
                self.inventory[item] = self.inventory.get(item, 0) + 10 # Add 10 items
                reward = 0.2 # Give a small reward for smart planning
                msg = f"Restocked 10 units of {item}."
            else:
                reward = -0.2
                msg = "Not enough budget to restock!"
        
        done = len(self.pending_orders) == 0 or self.steps_taken >= self.max_steps
        return self._get_obs(msg), reward, done, {}