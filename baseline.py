import os
import requests
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize the NEW Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_URL = "http://127.0.0.1:8000"

def play_game(level="easy"):
    print(f"\n🚀 --- Starting Gemini Baseline on {level.upper()} mode --- 🚀")
    
    # 1. Reset the Environment
    try:
        obs = requests.get(f"{BASE_URL}/reset/{level}").json()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is Uvicorn running in the other terminal?")
        return

    done = False
    step_count = 0
    
    while not done and step_count < 10:
        step_count += 1
        print(f"\n--- STEP {step_count} ---")
        print(f"📦 Inventory: {obs['inventory']}")
        print(f"💰 Budget: ${obs['budget']}")
        print(f"📋 Orders: {obs['pending_orders']}")
        
        # 2. Build the Prompt for Gemini
        prompt = f"""
        You are an AI warehouse manager. Your goal is to clear all pending orders efficiently.
        
        Current State:
        - Inventory: {obs['inventory']}
        - Budget: ${obs['budget']}
        - Pending Orders: {obs['pending_orders']}
        
        Available Actions:
        1. "ship" - Requires params: {{"order_id": "<id>", "carrier": "Standard"}}
        2. "restock" - Requires params: {{"item": "<item_name>"}}
        
        Analyze the state and choose the best action. 
        RESPOND ONLY IN STRICT JSON format matching this structure exactly:
        {{"command": "ship", "params": {{"order_id": "ORD-1", "carrier": "Standard"}}}}
        """
        
        # 3. Call Gemini using the new SDK
        print("🧠 Gemini is thinking...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        # Parse the JSON decision
        try:
            ai_action = json.loads(response.text)
            print(f"⚡ Gemini chose action: {ai_action}")
        except json.JSONDecodeError:
            print(f"❌ Gemini messed up the JSON output: {response.text}")
            break
        
        # 4. Send the action to your OpenEnv server
        step_res = requests.post(f"{BASE_URL}/step", json=ai_action).json()
        
        if "observation" not in step_res:
            print("❌ Error from server:", step_res)
            break
            
        obs = step_res["observation"]
        done = step_res["done"]
        print(f"⚖️ Reward received: {step_res['reward']}")
        
    # 5. Get the Final Hackathon Grade
    print("\n🏁 Game Over! Fetching Final Grade...")
    grade = requests.get(f"{BASE_URL}/grade").json()
    print(f"🏆 FINAL SCORE: {grade.get('final_score', 0.0)} / 1.0")

if __name__ == "__main__":
    play_game("hard")