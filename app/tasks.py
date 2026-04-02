import random

class LogisticsTasks:
    @staticmethod
    def get_task_setup(level: str):
        # Introduce randomness so the AI can't memorize the environment
        items = ["Electronics", "Clothing", "Medical", "Food"]
        random_item = random.choice(items)
        order_id = f"ORD-{random.randint(100, 999)}"
        
        if level == "easy":
            # Easy: Plentiful budget, item is already in stock
            budget = random.uniform(100.0, 150.0)
            inventory = {random_item: random.randint(5, 15)}
            orders = [{"id": order_id, "priority": "Standard", "item": random_item, "age": 0}]
            
        elif level == "medium":
            # Medium: Very tight budget, standard inventory
            budget = random.uniform(25.0, 40.0)
            inventory = {random_item: random.randint(5, 10), "Gadgets": 2}
            orders = [
                {"id": order_id, "priority": "VIP", "item": random_item, "age": 1},
                {"id": f"ORD-{random.randint(100, 999)}", "priority": "Standard", "item": "Gadgets", "age": 0}
            ]
            
        elif level == "hard":
            # Hard: The Stock-out Crisis (Zero inventory of the requested item)
            budget = random.uniform(80.0, 120.0)
            inventory = {random_item: 0} # AI MUST restock this first
            orders = [{"id": order_id, "priority": "VIP", "item": random_item, "age": 2}]
            
        else:
            # Default fallback
            budget = 100.0
            inventory = {"Electronics": 10}
            orders = [{"id": "ORD-000", "priority": "Standard", "item": "Electronics", "age": 0}]

        # Round budget to 2 decimal places for cleaner logs
        budget = round(budget, 2)
        
        return inventory, orders, budget