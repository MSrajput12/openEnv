# 📦 LogisticsFlow-OpenEnv

A real-world OpenEnv simulation for testing AI agents on Supply Chain Management, Budgeting, and Resource Allocation.

## 🎯 The Real-World Problem
Modern E-commerce fulfillment requires balancing strict budgets with customer satisfaction. AI agents must learn to prioritize high-value VIP orders, manage limited budgets, and anticipate stock-outs before they happen. 

This environment simulates a live warehouse dispatch system. Agents are not just playing a game; they are optimizing a simulated business.

## 🧠 Environment Features
* **Strict OpenEnv Compliance:** Fully typed `Action` and `Observation` models using Pydantic.
* **Continuous Partial Rewards:** Agents receive granular reward signals for every successful shipment, not just a binary score at the end.
* **Dynamic State Degradation:** Customer satisfaction drops as orders age in the queue, forcing the agent to act efficiently.
* **Deterministic Task Graders:** Built-in programmatic graders (Easy, Medium, Hard) that return a strict 0.0 to 1.0 score based on budget retention and order completion.

## 🚀 Tasks & Difficulty
1. **Easy (`/reset/easy`):** Basic capability test. Agent must ship existing inventory.
2. **Medium (`/reset/medium`):** Constraint optimization. Agent is given a severely limited budget and must choose the cheapest shipping carriers to survive.
3. **Hard (`/reset/hard`):** Multi-step reasoning. The agent is faced with a "Stock-out Crisis" (0 inventory). It must realize the shortage, execute a `restock` command, and *then* fulfill the orders.

## 🛠️ Action Space
* `ship`: Dispatch an order (Requires `order_id` and `carrier`).
* `restock`: Purchase more inventory to fulfill future orders (Requires `item`).

## 📊 Observation Space
* `inventory`: Current stock counts (e.g., `{"Electronics": 10}`).
* `pending_orders`: Queue of orders with priority levels and age.
* `budget`: Available capital.

## 🏃 How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `python -m uvicorn main:app --port 7860`
3. The environment will be available at `http://localhost:7860`