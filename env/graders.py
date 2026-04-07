# Advanced Multi-Factor Grader with Corridor Intelligence


def normalize(value, max_value):

    if max_value == 0:
        return 0.0

    return max(0.0, min(value / max_value, 1.0))


# -------------------------
# Completion Score
# -------------------------

def completion_score(env):

    if env.ambulance_pos == env.hospital_pos:

        return 1.0

    return 0.0


# -------------------------
# Efficiency Score
# -------------------------

def efficiency_score(env):

    step_ratio = env.current_step / env.max_steps

    return 1.0 - step_ratio


# -------------------------
# Safety Score
# -------------------------

def safety_score(env):

    max_collisions = env.max_steps / 5

    collision_penalty = normalize(
        env.collision_count,
        max_collisions
    )

    return 1.0 - collision_penalty


# -------------------------
# Signal Handling Score
# -------------------------

def signal_score(env):

    max_stops = env.max_steps / 3

    stop_penalty = normalize(
        env.signal_stops,
        max_stops
    )

    return 1.0 - stop_penalty


# -------------------------
# Communication Score
# -------------------------

def communication_score(env):

    expected_messages = env.num_vehicles * env.max_steps / 4

    usage = normalize(
        env.priority_messages,
        expected_messages
    )

    return usage


# -------------------------
# 🚑 Corridor Score (NEW)
# -------------------------

def corridor_score(env):

    if env.corridor_checks == 0:

        return 0.0

    return env.corridor_success / env.corridor_checks


# -------------------------
# Final Grade
# -------------------------

def grade(task_name, env):

    comp = completion_score(env)
    eff = efficiency_score(env)
    saf = safety_score(env)
    sig = signal_score(env)
    comm = communication_score(env)
    corr = corridor_score(env)

    final_score = (

        0.25 * comp +
        0.20 * eff +
        0.18 * saf +
        0.12 * sig +
        0.10 * comm +
        0.15 * corr

    )

    return round(final_score, 2)