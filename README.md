---
title: Ambclear
emoji: 🚑
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---
# 🚑 Ambclear — Intelligent Ambulance Clearance Environment

## 🧠 Problem Motivation
In real-world emergency scenarios, **ambulance delays due to traffic congestion** can directly impact patient survival.
This project models a **smart ambulance navigation system** where:
- Traffic dynamically reacts to emergency signals
- The ambulance must reach the hospital efficiently
- Safety, communication, and coordination are evaluated

---

## 🎯 Objective
Design and evaluate an agent that can:
- Reach the hospital **as quickly as possible**
- Avoid **collisions and unnecessary stops**
- Utilize **vehicle-to-vehicle (V2V) communication**
- Maintain a **clear corridor for emergency movement**

---

## 🌐 Environment Overview
A **7×7 grid-based traffic simulation** representing an urban road network.

### 🚦 Entities
| Value | Entity |
|------|--------|
| 0 | Empty Road |
| 1 | Vehicle |
| 2 | Ambulance |
| 3 | Hospital |
| 4 | Traffic Signal |

- Ambulance starts from left side → hospital is on right side
- Vehicles move dynamically (random + communication-aware)
- Signals introduce stochastic delays

---

## ⚙️ Action Space
Discrete actions:
| Action | Movement |
|--------|--------|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |

---

## 👁️ Observation Space
A **7×7 matrix** encoding full environment state:
- Ambulance position
- Vehicles
- Signals
- Hospital

---

## 📡 Key Innovation: BharatLink V2V Communication
A localized **device-to-device communication system** inspired by real-world mesh networking:
- Ambulance broadcasts priority signal within a **radius of 2 cells**
- Nearby vehicles that receive the signal **yield and clear the path**
- Vehicles outside the radius remain as obstacles
- A* pathfinding treats affected vehicles as **low-cost passable cells** (cost=2) instead of hard blocks
- Simulates real-world **emergency corridor formation**

```python
class BharatLinkComm:
    def __init__(self, radius=2):
        self.radius = radius

    def broadcast(self, ambulance_pos, vehicle_positions):
        # Returns vehicles within Manhattan distance of ambulance
        affected = []
        for vx, vy in vehicle_positions:
            if abs(ambulance_pos[0]-vx) + abs(ambulance_pos[1]-vy) <= self.radius:
                affected.append((vx, vy))
        return affected
```

👉 This enables **emergent cooperative behavior** — the corridor forms dynamically as the ambulance moves.

---

## 🚧 Corridor Intelligence
The system evaluates:
- Whether the path ahead is **clear of vehicles**
- How consistently a **corridor is maintained**

This mimics:
> Real-world "green corridor" strategies used in emergency response

---

## 🧪 Tasks (Difficulty Scaling)
| Task | Vehicles | Signals | Max Steps |
|------|----------|--------|----------|
| Easy | 2 | 1 | 40 |
| Medium | 5 | 2 | 50 |
| Hard | 8 | 3 | 60 |

✔ Increasing traffic density  
✔ Increasing coordination complexity  

---

## 🤖 Agent Architecture
The inference agent uses a **3-layer decision system**:

1. **BharatLink Broadcast** — identify yielding vehicles every step
2. **LLM (Primary)** — LLM navigates using grid state + BharatLink context + A* hint
3. **A* with BharatLink (Fallback)** — pathfinding that treats yielding vehicles as passable

---

## 📊 Multi-Factor Evaluation System
Unlike simple RL setups, this environment uses a **composite grading system**:

### 🔹 1. Completion (25%)
- Did the ambulance reach the hospital?

### 🔹 2. Efficiency (20%)
- How quickly was the goal achieved?

### 🔹 3. Safety (18%)
- Collision minimization

### 🔹 4. Signal Handling (12%)
- Stops due to traffic signals

### 🔹 5. Communication Usage (10%)
- Effective use of V2V signaling

### 🔹 6. Corridor Maintenance (15%)
- Path clearing consistency

---

## 📈 Baseline Scores

Scores produced by the baseline inference script (LLM + BharatLink + A*):

| Task | Success | Steps Taken | Max Steps | Score |
|------|---------|-------------|-----------|-------|
| Easy | ✅ | 6 | 40 | 0.955 |
| Medium | ✅ | 20 | 50 | 0.880 |
| Hard | ✅ | 10 | 60 | 0.950 |

> Scores are normalized in range [0.0, 1.0]. Higher = better.  
> BharatLink significantly improves hard task performance by enabling the ambulance to path through yielding vehicles.

---

## 🛠️ Setup & Usage

### Environment Variables Required
| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face API key |

### Run with Docker
```bash
docker build -t ambclear .
docker run -e API_BASE_URL=... -e MODEL_NAME=... -e HF_TOKEN=... ambclear
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Take action, returns obs/reward/done |
| `/state` | GET | Get current state |

### Example
```python
import requests

# Reset
obs = requests.post("http://localhost:7860/reset", json={"task": "easy"}).json()

# Step
result = requests.post("http://localhost:7860/step", json={"action": 3}).json()
print(result)  # {"observation": [...], "reward": 0.1, "done": false}
```

---

## 📁 Project Structure
```
├── inference.py          # Baseline agent (LLM + BharatLink + A*)
├── Dockerfile            # Container setup
├── openenv.yaml          # OpenEnv spec
├── requirements.txt      # Dependencies
├── pyproject.toml        # Package config
├── server/
│   └── app.py            # Server entry point
└── env/
    ├── ambulance_env.py  # Core environment
    ├── communication.py  # BharatLink V2V system
    ├── graders.py        # Task graders
    ├── tasks.py          # Task definitions
    └── models.py         # Typed models
```
