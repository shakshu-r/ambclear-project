import numpy as np
import random

from env.tasks import tasks
from env.communication import BharatLinkComm


class AmbclearEnv:

    def __init__(self, task_name="medium"):

        self.grid_size = 7

        # Load task config
        task = tasks[task_name]

        self.num_vehicles = int(task.vehicles)
        self.num_signals = int(task.signals)
        self.max_steps = int(task.max_steps)

        # BharatLink communication
        self.comm = BharatLinkComm(radius=2)

        self.reset()

    # -------------------------
    # RESET
    # -------------------------

    def reset(self):

        self.current_step = 0

        # Metrics
        self.collision_count = 0
        self.signal_stops = 0
        self.priority_messages = 0

        # Corridor metrics
        self.corridor_success = 0
        self.corridor_checks = 0

        self.ambulance_pos = [3, 0]
        self.hospital_pos = [3, 6]

        # -------------------------
        # Vehicles
        # -------------------------

        self.vehicle_positions = []

        while len(self.vehicle_positions) < self.num_vehicles:

            x = random.randint(0, 6)
            y = random.randint(0, 6)

            pos = [x, y]

            if (
                pos != self.ambulance_pos
                and pos != self.hospital_pos
                and pos not in self.vehicle_positions
            ):
                self.vehicle_positions.append(pos)

        # -------------------------
        # Signals
        # -------------------------

        self.signal_positions = []

        while len(self.signal_positions) < self.num_signals:

            x = random.randint(0, 6)
            y = random.randint(0, 6)

            pos = [x, y]

            if (
                pos != self.ambulance_pos
                and pos != self.hospital_pos
                and pos not in self.vehicle_positions
                and pos not in self.signal_positions
            ):
                self.signal_positions.append(pos)

        return self.state()

    # -------------------------
    # STATE
    # -------------------------

    def state(self):

        grid = np.zeros((self.grid_size, self.grid_size))

        ax, ay = self.ambulance_pos
        hx, hy = self.hospital_pos

        grid[ax][ay] = 2
        grid[hx][hy] = 3

        for vx, vy in self.vehicle_positions:
            grid[vx][vy] = 1

        for sx, sy in self.signal_positions:
            grid[sx][sy] = 4

        return grid

    # -------------------------
    # Corridor Check
    # -------------------------

    def check_corridor(self):

        ax, ay = self.ambulance_pos

        clear_cells = 0
        total_cells = 0

        for i in range(1, 3):

            ny = ay + i

            if ny <= 6:

                total_cells += 1

                if [ax, ny] not in self.vehicle_positions:
                    clear_cells += 1

        if total_cells > 0:

            if clear_cells == total_cells:
                self.corridor_success += 1

            self.corridor_checks += 1

    # -------------------------
    # STEP
    # -------------------------

    def step(self, action):

        self.current_step += 1

        ax, ay = self.ambulance_pos

        # -------------------------
        # BharatLink Broadcast
        # -------------------------

        priority_vehicles = self.comm.broadcast(
            self.ambulance_pos,
            self.vehicle_positions
        )

        self.priority_messages += len(priority_vehicles)

        # -------------------------
        # Move Vehicles
        # -------------------------

        new_vehicle_positions = []

        for vx, vy in self.vehicle_positions:

            dx = vx - ax
            dy = vy - ay

            if [vx, vy] in priority_vehicles:

                move_options = [
                    (1 if dx <= 0 else -1, 0),
                    (0, 1 if dy <= 0 else -1),
                    (0, 0)
                ]

            else:

                move_options = [
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                    (0, 0)
                ]

            move = random.choice(move_options)

            nx = vx + move[0]
            ny = vy + move[1]

            nx = max(0, min(nx, 6))
            ny = max(0, min(ny, 6))

            new_pos = [nx, ny]

            if new_pos not in new_vehicle_positions:
                new_vehicle_positions.append(new_pos)
            else:
                new_vehicle_positions.append([vx, vy])

        self.vehicle_positions = new_vehicle_positions

        # -------------------------
        # Move Ambulance
        # -------------------------

        new_ax = ax
        new_ay = ay

        if action == 0:
            new_ax -= 1
        elif action == 1:
            new_ax += 1
        elif action == 2:
            new_ay -= 1
        elif action == 3:
            new_ay += 1

        new_ax = max(0, min(new_ax, 6))
        new_ay = max(0, min(new_ay, 6))

        # -------------------------
        # Reward Logic
        # -------------------------

        old_distance = abs(ax - self.hospital_pos[0]) + abs(
            ay - self.hospital_pos[1]
        )

        new_distance = abs(new_ax - self.hospital_pos[0]) + abs(
            new_ay - self.hospital_pos[1]
        )

        reward = 0.0
        done = False

        # Signal Check
        if [new_ax, new_ay] in self.signal_positions:

            signal_state = random.choice(["red", "green"])

            if signal_state == "red":
                self.signal_stops += 1
                new_ax, new_ay = ax, ay
                reward -= 0.05

        # Collision
        if [new_ax, new_ay] in self.vehicle_positions:

            self.collision_count += 1
            reward -= 0.5

        # Hospital
        elif [new_ax, new_ay] == self.hospital_pos:

            reward += 1.0
            done = True

        else:

            if new_distance < old_distance:
                reward += 0.1
            else:
                reward -= 0.02

        # Step limit
        if self.current_step >= self.max_steps:
            done = True

        self.ambulance_pos = [new_ax, new_ay]

        # Corridor check
        self.check_corridor()

        # -------------------------
        # INFO (OpenEnv Compliance)
        # -------------------------

        info = {
            "collisions": self.collision_count,
            "signal_stops": self.signal_stops,
            "priority_messages": self.priority_messages,
            "corridor_success": self.corridor_success
        }

        return self.state(), reward, done, info

    # -------------------------
    # RENDER
    # -------------------------

    def render(self):
        print(self.state())