from flask import Flask
import random
import numpy as np
import os

from openai import OpenAI
from env.ambulance_env import AmbclearEnv
from env.graders import grade

# -------------------------
# ENV VARIABLES
# -------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-7B-Instruct"
)

HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

print("HF_TOKEN loaded:", bool(HF_TOKEN), flush=True)

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing. Add it in Secrets.")

# -------------------------
# OPENAI CLIENT
# -------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# -------------------------
# FLASK APP
# -------------------------

app = Flask(__name__)

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "ambclear"

# -------------------------
# SEED
# -------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# GLOBAL LLM STATE
# -------------------------

LLM_AVAILABLE = True

# -------------------------
# LLM FUNCTION (SAFE)
# -------------------------

def query_llm(prompt: str):
    global LLM_AVAILABLE

    if not LLM_AVAILABLE:
        return None

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY one number: 0, 1, 2, or 3"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=5
        )

        output = response.choices[0].message.content.strip()

        for ch in output:
            if ch in "0123":
                return int(ch)

        return None

    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)

        # Disable LLM if quota is exhausted
        if "402" in str(e) or "credits" in str(e).lower():
            print("[LLM DISABLED] switching to local policy only", flush=True)
            LLM_AVAILABLE = False

        return None

# -------------------------
# SMART POLICY (MAIN AGENT)
# -------------------------

def get_action(env):

    ax, ay = env.ambulance_pos
    hx, hy = env.hospital_pos

    move_map = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1)
    }

    def is_safe(nx, ny):
        return [nx, ny] not in env.vehicle_positions

    # -------------------------
    # GREEDY DISTANCE POLICY (PRIMARY)
    # -------------------------

    candidates = []

    for action, (dx, dy) in move_map.items():
        nx = max(0, min(ax + dx, 6))
        ny = max(0, min(ay + dy, 6))

        dist = abs(nx - hx) + abs(ny - hy)

        if is_safe(nx, ny):
            candidates.append((dist, action))

    if candidates:
        candidates.sort()
        return candidates[0][1]

    # -------------------------
    # LLM FALLBACK
    # -------------------------

    prompt = f"""
Ambulance: ({ax},{ay})
Hospital: ({hx},{hy})
Obstacles: {env.vehicle_positions}
"""

    llm_action = query_llm(prompt)

    if llm_action is not None:
        return llm_action

    # -------------------------
    # SAFE RANDOM FALLBACK
    # -------------------------

    safe_actions = []

    for action, (dx, dy) in move_map.items():
        nx = max(0, min(ax + dx, 6))
        ny = max(0, min(ay + dy, 6))

        if is_safe(nx, ny):
            safe_actions.append(action)

    if safe_actions:
        return random.choice(safe_actions)

    return random.randint(0, 3)

# -------------------------
# RUN TASK
# -------------------------

def run_task(task_name):

    env = AmbclearEnv(task_name)
    state = env.reset()

    rewards = []
    steps = 0
    done = False

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while not done:

            steps += 1
            action = get_action(env)

            result = env.step(action)

            if len(result) == 3:
                state, reward, done = result
            else:
                state, reward, done, _ = result

            rewards.append(reward)

            print(
                f"[STEP] step={steps} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            if steps >= env.max_steps:
                break

        score = grade(task_name, env)
        success = score >= 0.5

    except Exception as e:
        print(f"[STEP] step={steps} action=error reward=0 done=true error={e}", flush=True)
        score = 0.0
        success = False

    print(
        f"[END] success={success} steps={steps} score={score:.3f} rewards={rewards}",
        flush=True
    )

# -------------------------
# MAIN LOOP
# -------------------------

def run_inference():
    for task in TASKS:
        run_task(task)

has_run = False

@app.route("/")
def home():
    global has_run
    if not has_run:
        run_inference()
        has_run = True
    return "<pre>Inference completed</pre>"

# -------------------------
# ENTRYPOINT
# -------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)