import os
import random
import numpy as np
from openai import OpenAI
from env.ambulance_env import AmbclearEnv
from env.graders import grade

# -------------------------
# ENV VARIABLES
# -------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "ambclear-env"

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing.")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LLM_AVAILABLE = True

# -------------------------
# LOGGING
# -------------------------

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# -------------------------
# LLM
# -------------------------

def query_llm(prompt: str):
    global LLM_AVAILABLE
    if not LLM_AVAILABLE:
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return ONLY one number: 0, 1, 2, or 3"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        output = response.choices[0].message.content.strip()
        for ch in output:
            if ch in "0123":
                return int(ch)
        return None
    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        if "402" in str(e) or "credit" in str(e).lower():
            LLM_AVAILABLE = False
        return None

# -------------------------
# POLICY
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

    candidates = []
    for action, (dx, dy) in move_map.items():
        nx = max(0, min(ax + dx, 6))
        ny = max(0, min(ay + dy, 6))
        if is_safe(nx, ny):
            dist = abs(nx - hx) + abs(ny - hy)
            candidates.append((dist, action))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    prompt = f"Ambulance: ({ax},{ay}) | Hospital: ({hx},{hy}) | Obstacles: {env.vehicle_positions}"
    llm_action = query_llm(prompt)
    if llm_action in [0, 1, 2, 3]:
        return llm_action

    safe_actions = []
    for action, (dx, dy) in move_map.items():
        nx = max(0, min(ax + dx, 6))
        ny = max(0, min(ay + dy, 6))
        if is_safe(nx, ny):
            safe_actions.append(action)

    return random.choice(safe_actions) if safe_actions else random.randint(0, 3)

# -------------------------
# MAIN
# -------------------------

def main():
    env = AmbclearEnv(TASK_NAME)
    env.reset()

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    done = False
    success = False
    score = 0.0

    try:
        for step in range(1, env.max_steps + 1):
            if done:
                break

            action = get_action(env)
            result = env.step(action)

            if len(result) == 3:
                state, reward, done = result
            else:
                state, reward, done, _ = result

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=str(action), reward=reward, done=done)

            if done:
                break

        score = grade(TASK_NAME, env)
        success = env.ambulance_pos == env.hospital_pos

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()