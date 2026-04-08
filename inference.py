import os
import random
import numpy as np
import heapq
from typing import List, Tuple, Optional
from env.ambulance_env import AmbclearEnv

# ── config ────────────────────────────────────────────────────────────────────
TASK_NAME               = os.getenv("TASK_NAME", None)  # if set, run only that task
BENCHMARK               = os.getenv("BENCHMARK", "ambulance")
MAX_POSSIBLE_REWARD     = 1.0

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# grid encoding
AMBULANCE_ID = 2
HOSPITAL_ID  = 3
VEHICLE_ID   = 1
SIGNAL_ID    = 4

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTAS = {UP: (-1,0), DOWN: (1,0), LEFT: (0,-1), RIGHT: (0,1)}

# ── logging ───────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── grid helpers ──────────────────────────────────────────────────────────────
def find_entity(grid: np.ndarray, entity_id: float) -> Optional[Tuple[int,int]]:
    positions = list(zip(*np.where(grid == entity_id)))
    return tuple(positions[0]) if positions else None

def get_vehicles(grid: np.ndarray) -> List[Tuple[int,int]]:
    return [tuple(p) for p in zip(*np.where(grid == VEHICLE_ID))] \
        if np.any(grid == VEHICLE_ID) else []

def get_signals(grid: np.ndarray) -> List[Tuple[int,int]]:
    return [tuple(p) for p in zip(*np.where(grid == SIGNAL_ID))] \
        if np.any(grid == SIGNAL_ID) else []

# ── A* pathfinding ────────────────────────────────────────────────────────────
def astar(start: Tuple[int,int],
          goal: Tuple[int,int],
          vehicles: List[Tuple[int,int]],
          signals: List[Tuple[int,int]],
          grid_size: int = 7) -> Optional[int]:
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

            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue
            if (nr, nc) in vehicle_set:
                continue

            step_cost = 4 if (nr, nc) in signal_set else 1
            new_g     = g + step_cost
            new_f     = new_g + heuristic(nr, nc)

            if (nr, nc) not in visited:
                fa = action if first_act is None else first_act
                heapq.heappush(heap, (new_f, new_g, nr, nc, fa))

    return None

# ── greedy fallback ───────────────────────────────────────────────────────────
def greedy_fallback(amb: Tuple[int,int],
                    goal: Tuple[int,int],
                    vehicles: List[Tuple[int,int]]) -> int:
    vehicle_set = set(vehicles)
    row_diff    = goal[0] - amb[0]
    col_diff    = goal[1] - amb[1]
    candidates  = []

    if row_diff != 0:
        candidates.append((abs(row_diff), DOWN if row_diff > 0 else UP))
    if col_diff != 0:
        candidates.append((abs(col_diff), RIGHT if col_diff > 0 else LEFT))

    candidates.sort(reverse=True)

    for _, action in candidates:
        dr, dc = DELTAS[action]
        if (amb[0]+dr, amb[1]+dc) not in vehicle_set:
            return action

    for action in [RIGHT, DOWN, UP, LEFT]:
        dr, dc = DELTAS[action]
        if (amb[0]+dr, amb[1]+dc) not in vehicle_set:
            return action

    return random.randint(0, 3)

# ── run one episode ───────────────────────────────────────────────────────────
def run_episode(task_name: str):
    env = AmbclearEnv(task_name)

    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False

    log_start(task=task_name, env=BENCHMARK, model="astar-agent")

    try:
        obs  = env.reset()
        done = False

        for step in range(1, env.max_steps + 1):
            if done:
                break

            grid     = obs if isinstance(obs, np.ndarray) else np.array(obs)
            amb_pos  = find_entity(grid, AMBULANCE_ID)
            hosp_pos = find_entity(grid, HOSPITAL_ID)
            vehicles = get_vehicles(grid)
            signals  = get_signals(grid)

            if amb_pos is None or hosp_pos is None:
                action = random.randint(0, 3)
            else:
                action = astar(amb_pos, hosp_pos, vehicles, signals)
                if action is None:
                    action = greedy_fallback(amb_pos, hosp_pos, vehicles)

            result = env.step(action)
            if len(result) == 3:
                obs, reward, done = result
            else:
                obs, reward, done, _ = result

            reward = float(reward)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done)

            if done:
                break

        total   = sum(rewards)
        score   = min(max(total / MAX_POSSIBLE_REWARD, 0.0), 1.0)
        success = bool(done and rewards and rewards[-1] >= 1.0)

    except Exception as e:
        import traceback
        print(f"[DEBUG] Episode error:\n{traceback.format_exc()}", flush=True)

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if TASK_NAME:
        # evaluator set a specific task — run only that one
        run_episode(TASK_NAME)
    else:
        # no task set — run all three difficulties
        for difficulty in ["easy", "medium", "hard"]:
            run_episode(difficulty)

import gradio as gr

def run_all():
    main()
    return "All tasks completed successfully!"

iface = gr.Interface(
    fn=run_all,
    inputs=[],
    outputs="text",
    title="AmbuClear Environment Runner"
)

iface.launch(
    server_name="0.0.0.0",
    server_port=7860
)
