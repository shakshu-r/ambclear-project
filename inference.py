from flask import Flask
import random
import numpy as np
import os

from env.ambulance_env import AmbclearEnv  # ensure this matches your class name
from env.graders import grade

app = Flask(__name__)

# -------------------------
# ENV VARIABLES
# -------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-agent")
HF_TOKEN = os.getenv("HF_TOKEN")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "ambclear"

# -------------------------
# SEED (REPRODUCIBILITY)
# -------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# SMART POLICY
# -------------------------

def get_action(env):

    ax, ay = env.ambulance_pos
    hx, hy = env.hospital_pos

    preferred_moves = []

    if hy > ay:
        preferred_moves.append(3)
    if hy < ay:
        preferred_moves.append(2)
    if hx > ax:
        preferred_moves.append(1)
    if hx < ax:
        preferred_moves.append(0)

    move_map = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1)
    }

    # Try preferred safe moves
    for action in preferred_moves:
        dx, dy = move_map[action]
        nx = max(0, min(ax + dx, 6))
        ny = max(0, min(ay + dy, 6))

        if [nx, ny] not in env.vehicle_positions:
            return action

    # Try any safe move
    for action, (dx, dy) in move_map.items():
        nx = max(0, min(ax + dx, 6))
        ny = max(0, min(ay + dy, 6))

        if [nx, ny] not in env.vehicle_positions:
            return action

    # Last fallback
    return random.randint(0, 3)

# -------------------------
# RUN SINGLE TASK
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

            # Handle both 3 or 4 return values
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
        success = str(score >= 0.5).lower()

    except Exception as e:
        print(
            f"[STEP] step={steps} action=error reward=0.00 done=true error={str(e)}",
            flush=True
        )
        score = 0.0
        success = "false"

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={success} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# -------------------------
# MAIN INFERENCE
# -------------------------

def run_inference():
    for task in TASKS:
        run_task(task)

# -------------------------
# SINGLE RUN GUARD
# -------------------------

has_run = False

@app.route("/")
def home():
    global has_run
    if not has_run:
        run_inference()
        has_run = True
    return "<pre>Inference completed</pre>"

# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)