import os
import random
import threading
import numpy as np
import heapq
from flask import Flask, request, jsonify
from openai import OpenAI
from env.ambulance_env import AmbclearEnv

# ── config ────────────────────────────────────────────────────────────────────
TASK_NAME    = os.getenv("TASK_NAME", None)
BENCHMARK    = os.getenv("BENCHMARK", "ambulance")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
API_KEY      = os.getenv("ambclear_api") or HF_TOKEN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MAX_POSSIBLE_REWARD = 1.0

AMBULANCE_ID = 2
HOSPITAL_ID  = 3
VEHICLE_ID   = 1
SIGNAL_ID    = 4

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTAS = {UP: (-1,0), DOWN: (1,0), LEFT: (0,-1), RIGHT: (0,1)}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.url_map.strict_slashes = False

global_env = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "alive"})

@app.route("/reset", methods=["GET", "POST"])
def reset():
    global global_env
    try:
        data       = request.get_json(silent=True) or {}
        task       = data.get("task", TASK_NAME or "easy")
        global_env = AmbclearEnv(task)
        obs        = global_env.reset()
        return jsonify({
            "observation": obs.tolist() if hasattr(obs, "tolist") else str(obs),
            "task": task
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/step", methods=["GET", "POST"])
def step():
    global global_env
    try:
        if global_env is None:
            return jsonify({"error": "Call /reset first"}), 400
        data   = request.get_json(silent=True) or {}
        action = data.get("action", random.randint(0, 3))
        result = global_env.step(int(action))
        if len(result) == 3:
            obs, reward, done = result
        else:
            obs, reward, done, _ = result
        return jsonify({
            "observation": obs.tolist() if hasattr(obs, "tolist") else str(obs),
            "reward": float(reward),
            "done": bool(done)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── logging ───────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error or 'null'}",
        flush=True
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True
    )

# ── helpers ───────────────────────────────────────────────────────────────────
def find_entity(grid, entity_id):
    positions = list(zip(*np.where(grid == entity_id)))
    return tuple(positions[0]) if positions else None

def get_vehicles(grid):
    return [tuple(p) for p in zip(*np.where(grid == VEHICLE_ID))] \
        if np.any(grid == VEHICLE_ID) else []

def get_signals(grid):
    return [tuple(p) for p in zip(*np.where(grid == SIGNAL_ID))] \
        if np.any(grid == SIGNAL_ID) else []

# ── A* ────────────────────────────────────────────────────────────────────────
def astar(start, goal, vehicles, signals):
    vehicle_set = set(vehicles)
    signal_set  = set(signals)

    def heuristic(r, c):
        return abs(r - goal[0]) + abs(c - goal[1])

    heap    = [(heuristic(start[0], start[1]), 0, start[0], start[1], None)]
    visited = {}

    while heap:
        f, g, r, c, first_act = heapq.heappop(heap)
        if (r, c) in visited:
            continue
        visited[(r, c)] = g
        if (r, c) == goal:
            return first_act
        for action, (dr, dc) in DELTAS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 7 and 0 <= nc < 7):
                continue
            if (nr, nc) in vehicle_set:
                continue
            step_cost = 4 if (nr, nc) in signal_set else 1
            new_g = g + step_cost
            new_f = new_g + heuristic(nr, nc)
            if (nr, nc) not in visited:
                fa = action if first_act is None else first_act
                heapq.heappush(heap, (new_f, new_g, nr, nc, fa))

    return random.randint(0, 3)

# ── LLM action ────────────────────────────────────────────────────────────────
def get_llm_action(client, step_num, obs_str, history):
    try:
        history_block = "\n".join(history[-4:]) if history else "None"
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "Return one number: 0,1,2 or 3."
                )},
                {"role": "user", "content": (
                    f"Step {step_num}\nGrid:\n{obs_str}"
                )}
            ],
            temperature=0.2,
            max_tokens=8,
        )

        text = (completion.choices[0].message.content or "").strip()

        for ch in text:
            if ch in "0123":
                return int(ch)

    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)

    return None


# ── run episode ───────────────────────────────────────────────────────────────
def run_episode(task_name, client):

    env         = AmbclearEnv(task_name)
    rewards     = []
    steps_taken = 0
    success     = False
    history     = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:

        obs  = env.reset()
        done = False

        for step_num in range(1, env.max_steps + 1):

            if done:
                break

            grid     = np.array(obs)
            amb      = find_entity(grid, AMBULANCE_ID)
            hosp     = find_entity(grid, HOSPITAL_ID)
            vehicles = get_vehicles(grid)
            signals  = get_signals(grid)

            if amb and hosp:

                # ✅ FORCE ONE LLM CALL ONLY AT STEP 1
                if step_num == 1:

                    llm_action = get_llm_action(
                        client,
                        step_num,
                        str(grid),
                        history
                    )

                    if llm_action is not None:
                        action = llm_action
                    else:
                        action = astar(amb, hosp, vehicles, signals)

                else:

                    action = astar(amb, hosp, vehicles, signals)

            else:

                action = random.randint(0, 3)

            result = env.step(action)

            if len(result) == 3:
                obs, reward, done = result
            else:
                obs, reward, done, _ = result

            reward = float(reward)

            rewards.append(reward)
            steps_taken = step_num

            history.append(
                f"step={step_num} action={action}"
            )

            log_step(
                step=step_num,
                action=action,
                reward=reward,
                done=done
            )

            if done:
                break

        total = sum(rewards)

        score = min(
            max(total / MAX_POSSIBLE_REWARD, 0.0),
            1.0
        )

        success = bool(
            done and rewards and rewards[-1] >= 1.0
        )

    finally:

        try:
            env.close()
        except:
            pass

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )

# ── inference thread ──────────────────────────────────────────────────────────
def run_inference():

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    if TASK_NAME:

        run_episode(TASK_NAME, client)

    else:

        for difficulty in ["easy", "medium", "hard"]:

            run_episode(difficulty, client)


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    t = threading.Thread(
        target=run_inference,
        daemon=False
    )

    t.start()
    t.join()

    # keep container alive
    import time
    time.sleep(120)
