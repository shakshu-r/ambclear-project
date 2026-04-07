class BharatLinkComm:

    def __init__(self, radius=2):

        # Communication range
        self.radius = radius

    def broadcast(self, ambulance_pos, vehicle_positions):

        ax, ay = ambulance_pos

        affected_vehicles = []

        for vx, vy in vehicle_positions:

            distance = abs(ax - vx) + abs(ay - vy)

            if distance <= self.radius:

                affected_vehicles.append([vx, vy])

        return affected_vehicles