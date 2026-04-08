import os
import random
import numpy as np
import heapq
from typing import List, Tuple, Optional

from env.ambulance_env import AmbclearEnv

# ── config ─────────────────────────────────────

TASK_NAME = os.getenv("TASK_NAME", None)
BENCHMARK = os.getenv("BENCHMARK", "ambulance")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MAX_POSSIBLE_REWARD = 1.0

# grid encoding
AMBULANCE_ID = 2
HOSPITAL_ID  = 3
VEHICLE_ID   = 1
SIGNAL_ID    = 4

UP, DOWN, LEFT, RIGHT = 0,1,2,3

DELTAS = {
    UP: (-1,0),
    DOWN: (1,0),
    LEFT: (0,-1),
    RIGHT: (0,1)
}

# ── logging (MANDATORY FORMAT) ─────────────────

def log_start(task, env, model):
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True
    )

def log_step(step, action, reward, done, error=None):

    error_val = error if error else "null"

    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error_val}",
        flush=True
    )

def log_end(success, steps, score, rewards):

    rewards_str = ",".join(
        f"{r:.2f}" for r in rewards
    )

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True
    )

# ── helpers ─────────────────────────────────────

def find_entity(grid, entity_id):

    positions = list(zip(*np.where(grid == entity_id)))

    return tuple(positions[0]) if positions else None


def get_vehicles(grid):

    if np.any(grid == VEHICLE_ID):
        return [tuple(p) for p in zip(*np.where(grid == VEHICLE_ID))]

    return []


def get_signals(grid):

    if np.any(grid == SIGNAL_ID):
        return [tuple(p) for p in zip(*np.where(grid == SIGNAL_ID))]

    return []


# ── A* ─────────────────────────────────────────

def astar(start, goal, vehicles, signals):

    vehicle_set = set(vehicles)
    signal_set  = set(signals)

    def heuristic(r,c):
        return abs(r-goal[0]) + abs(c-goal[1])

    heap = [(heuristic(start[0],start[1]),0,start[0],start[1],None)]

    visited = {}

    while heap:

        f,g,r,c,first_act = heapq.heappop(heap)

        if (r,c) in visited:
            continue

        visited[(r,c)] = g

        if (r,c) == goal:
            return first_act

        for action,(dr,dc) in DELTAS.items():

            nr,nc = r+dr , c+dc

            if not (0<=nr<7 and 0<=nc<7):
                continue

            if (nr,nc) in vehicle_set:
                continue

            step_cost = 4 if (nr,nc) in signal_set else 1

            new_g = g + step_cost
            new_f = new_g + heuristic(nr,nc)

            if (nr,nc) not in visited:

                fa = action if first_act is None else first_act

                heapq.heappush(
                    heap,
                    (new_f,new_g,nr,nc,fa)
                )

    return random.randint(0,3)


# ── run episode ───────────────────────────────

def run_episode(task_name):

    env = AmbclearEnv(task_name)

    rewards = []
    steps_taken = 0
    success = False

    log_start(
        task=task_name,
        env=BENCHMARK,
        model="astar-agent"
    )

    try:

        obs = env.reset()

        done = False

        for step in range(1, env.max_steps+1):

            if done:
                break

            grid = np.array(obs)

            amb = find_entity(grid,AMBULANCE_ID)
            hosp = find_entity(grid,HOSPITAL_ID)

            vehicles = get_vehicles(grid)
            signals  = get_signals(grid)

            if amb and hosp:

                action = astar(
                    amb,
                    hosp,
                    vehicles,
                    signals
                )

            else:

                action = random.randint(0,3)

            result = env.step(action)

            if len(result)==3:
                obs,reward,done = result
            else:
                obs,reward,done,_ = result

            reward=float(reward)

            rewards.append(reward)

            steps_taken=step

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=done
            )

            if done:
                break

        total=sum(rewards)

        score=min(
            max(total/MAX_POSSIBLE_REWARD,0.0),
            1.0
        )

        success=bool(
            done and rewards and rewards[-1]>=1.0
        )

    except Exception as e:

        import traceback

        print(
            traceback.format_exc(),
            flush=True
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


# ── main ─────────────────────────────────────

def main():

    if TASK_NAME:

        run_episode(TASK_NAME)

    else:

        for difficulty in ["easy","medium","hard"]:

            run_episode(difficulty)


if __name__ == "__main__":
    main()

    ''' Keep container alive briefly so validator can read logs
    import time
    time.sleep(120)'''
