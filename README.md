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

## 📡 Key Innovation: BharatLink Communication

A localized **V2V communication system**:

- Ambulance broadcasts priority within a radius
- Nearby vehicles adjust movement to clear path
- Simulates real-world **emergency corridor formation**

👉 This enables **emergent cooperative behavior**

---

## 🚧 Corridor Intelligence (Advanced Feature)

The system evaluates:

- Whether the path ahead is **clear of vehicles**
- How consistently a **corridor is maintained**

This mimics:
> Real-world “green corridor” strategies used in emergency response

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

## 📈 Final Score

All metrics are normalized:

```text
Final Score ∈ [0.0, 1.0]