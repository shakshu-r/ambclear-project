import os
import random
import threading
import numpy as np
import heapq
from flask import Flask, request, jsonify
from openai import OpenAI
from env.ambulance_env import AmbclearEnv
from env.graders import grade

# ── config ────────────────────────────────────────────────────────────────────
TASK_NAME    = os.getenv("TASK_NAME", None)
BENCHMARK    = os.getenv("BENCHMARK", "ambulance")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
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

# ── BharatLink V2V Communication ──────────────────────────────────────────────
class BharatLinkComm:
    def __init__(self, radius=2):
        self.radius = radius

    def broadcast(self, ambulance_pos, vehicle_positions):
        ax, ay = ambulance_pos
        affected = []
        for vx, vy in vehicle_positions:
            distance = abs(ax - vx) + abs(ay - vy)
            if distance <= self.radius:
                affected.append((vx, vy))
        return affected

# ── ASCII Visualizer ──────────────────────────────────────────────────────────
def render_ascii(grid, affected_vehicles=None):
    affected_set = set(affected_vehicles) if affected_vehicles else set()
    symbols = {
        0: ' . ',
        1: '[V]',
        2: '[A]',
        3: '[H]',
        4: '[S]',
    }
    lines = []
    lines.append("  +" + "---+" * 7)
    for r in range(7):
        row = "  |"
        for c in range(7):
            val = int(grid[r][c])
            if val == 1 and (r, c) in affected_set:
                row += '[~]'  # yielding vehicle
            else:
                row += symbols.get(val, ' ? ')
            row += '|'
        lines.append(row)
        lines.append("  +" + "---+" * 7)
    return "\n".join(lines)

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

# ── grid helpers ──────────────────────────────────────────────────────────────
def find_entity(grid, entity_id):
    positions = list(zip(*np.where(grid == entity_id)))
    return tuple(positions[0]) if positions else None

def get_vehicles(grid):
    return [tuple(p) for p in zip(*np.where(grid == VEHICLE_ID))] \
        if np.any(grid == VEHICLE_ID) else []

def get_signals(grid):
    return [tuple(p) for p in zip(*np.where(grid == SIGNAL_ID))] \
        if np.any(grid == SIGNAL_ID) else []

# ── A* with BharatLink awareness ──────────────────────────────────────────────
def astar(start, goal, vehicles, signals, affected_vehicles=None):
    vehicle_set  = set(vehicles)
    signal_set   = set(signals)
    affected_set = set(affected_vehicles) if affected_vehicles else set()

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
            if (nr, nc) in vehicle_set and (nr, nc) not in affected_set:
                continue
            if (nr, nc) in affected_set:
                step_cost = 2
            elif (nr, nc) in signal_set:
                step_cost = 4
            else:
                step_cost = 1
            new_g = g + step_cost
            new_f = new_g + heuristic(nr, nc)
            if (nr, nc) not in visited:
                fa = action if first_act is None else first_act
                heapq.heappush(heap, (new_f, new_g, nr, nc, fa))

    return random.randint(0, 3)

# ── LLM action (PRIMARY - called every step) ──────────────────────────────────
def get_llm_action(client, step_num, grid, amb, hosp, vehicles, signals,
                   history, affected_vehicles):
    try:
        history_block = "\n".join(history[-4:]) if history else "None"
        astar_hint    = astar(amb, hosp, vehicles, signals, affected_vehicles)

        prompt = (
            f"You control ambulance on a 7x7 grid.\n"
            f"Grid (0=empty,1=vehicle,2=ambulance,3=hospital,4=signal):\n{grid}\n"
            f"Ambulance at: {amb}\n"
            f"Hospital at: {hosp}\n"
            f"Vehicles at: {vehicles}\n"
            f"Vehicles that received BharatLink signal (will yield): {affected_vehicles}\n"
            f"Step: {step_num}\n"
            f"Recent history: {history_block}\n"
            f"Suggested action: {astar_hint}\n"
            f"Actions: 0=UP 1=DOWN 2=LEFT 3=RIGHT\n"
            f"Reply with ONLY one digit 0,1,2 or 3."
        )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "You are an ambulance navigation agent. "
                    "Vehicles that received BharatLink signal will yield — "
                    "you can move through them. "
                    "Always reply with exactly one digit: 0, 1, 2, or 3."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=8,
            timeout=10,
        )
        text = (completion.choices[0].message.content or "").strip()
        for ch in text:
            if ch in "0123":
                return int(ch)
    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)
    return None

# ── run one episode ───────────────────────────────────────────────────────────
def run_episode(task_name, client):
    env         = AmbclearEnv(task_name)
    comm        = BharatLinkComm(radius=2)
    rewards     = []
    steps_taken = 0
    success     = False
    score       = 0.0
    history     = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs  = env.reset()
        done = False

        # Print initial grid
        print(f"\n[GRID] Initial state — task={task_name}", flush=True)
        print(render_ascii(obs), flush=True)

        for step_num in range(1, env.max_steps + 1):
            if done:
                break

            grid     = np.array(obs)
            amb      = find_entity(grid, AMBULANCE_ID)
            hosp     = find_entity(grid, HOSPITAL_ID)
            vehicles = get_vehicles(grid)
            signals  = get_signals(grid)

            if amb and hosp:
                affected = comm.broadcast(amb, vehicles)
                print(f"[BHARATLINK] step={step_num} affected={affected}", flush=True)

                action = get_llm_action(
                    client, step_num, grid,
                    amb, hosp, vehicles, signals,
                    history, affected
                )
                if action is None:
                    action = astar(amb, hosp, vehicles, signals, affected)
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
            history.append(f"step={step_num} action={action} reward={reward:.2f}")

            log_step(step=step_num, action=action, reward=reward, done=done)

            # Print grid every 5 steps and on completion
            if step_num % 5 == 0 or done:
                print(f"\n[GRID] After step {step_num}:", flush=True)
                print(render_ascii(np.array(obs), affected), flush=True)

            if done:
                break

        # ── use real grader ───────────────────────────────────────────────────
        score   = grade(task_name, env)
        success = bool(done and rewards and rewards[-1] >= 1.0)

        # ── print detailed metrics ────────────────────────────────────────────
        print(f"\n[METRICS] task={task_name}", flush=True)
        print(f"  collisions    : {env.collision_count}", flush=True)
        print(f"  signal_stops  : {env.signal_stops}", flush=True)
        print(f"  priority_msgs : {env.priority_messages}", flush=True)
        print(f"  corridor      : {env.corridor_success}/{env.corridor_checks}", flush=True)
        print(f"  score         : {score}", flush=True)

    except Exception:
        import traceback
        print(traceback.format_exc(), flush=True)

    finally:
        try:
            env.close()
        except:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ── inference ─────────────────────────────────────────────────────────────────
def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    if TASK_NAME:
        run_episode(TASK_NAME, client)
    else:
        for difficulty in ["easy", "medium", "hard"]:
            run_episode(difficulty, client)

# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=run_inference, daemon=False)
    t.start()
    t.join()
    print("Inference complete.", flush=True)
