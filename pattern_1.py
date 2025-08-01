import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
num_drones = 10
width, height = 100, 100
max_speed = 2
neighbor_radius = 15
separation_distance = 5
formation_type = "v"  # Options: "line", "circle", "v"

# Initialization
positions = np.random.rand(num_drones, 2) * [width, height]
velocities = (np.random.rand(num_drones, 2) - 0.5) * max_speed

def limit_speed(v, max_speed):
    speed = np.linalg.norm(v)
    if speed > max_speed:
        return v / speed * max_speed
    return v

def get_formation_targets(center, spacing):
    if formation_type == "line":
        return np.array([[center[0] + (i - num_drones / 2) * spacing, center[1]] for i in range(num_drones)])
    elif formation_type == "circle":
        radius = spacing * num_drones / (2 * np.pi)
        return np.array([[center[0] + radius * np.cos(2 * np.pi * i / num_drones),
                          center[1] + radius * np.sin(2 * np.pi * i / num_drones)]
                         for i in range(num_drones)])
    elif formation_type == "v":
        return np.array([[center[0] + (i - num_drones//2) * spacing, 
                          center[1] + abs(i - num_drones//2) * spacing * 0.5]
                         for i in range(num_drones)])
    else:
        return positions

def update(frame_num):
    global positions, velocities
    new_positions = positions.copy()
    new_velocities = velocities.copy()
    center = np.array([width/2, height/2])
    targets = get_formation_targets(center, 5)

    for i in range(num_drones):
        pos_i = positions[i]
        vel_i = velocities[i]

        # Initialize rule vectors
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        separation = np.zeros(2)
        count = 0

        for j in range(num_drones):
            if i == j:
                continue
            dist = np.linalg.norm(positions[j] - pos_i)
            if dist < neighbor_radius:
                alignment += velocities[j]
                cohesion += positions[j]
                if dist < separation_distance:
                    separation -= (positions[j] - pos_i) / (dist + 1e-6)
                count += 1

        if count > 0:
            alignment /= count
            alignment = limit_speed(alignment - vel_i, max_speed)

            cohesion /= count
            cohesion = limit_speed(cohesion - pos_i, max_speed)

            separation = limit_speed(separation, max_speed)

        # Formation attraction
        to_target = limit_speed(targets[i] - pos_i, max_speed)

        # Weighting the behaviors
        velocity_update = (0.5 * alignment +
                           0.3 * cohesion +
                           0.7 * separation +
                           0.5 * to_target)

        new_velocities[i] = limit_speed(vel_i + velocity_update, max_speed)
        new_positions[i] = pos_i + new_velocities[i]

    positions = new_positions
    velocities = new_velocities
    scat.set_offsets(positions)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
scat = ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=50)

ani = animation.FuncAnimation(fig, update, interval=100)
plt.title(f"Swarm Drone Formation: {formation_type.upper()}")
plt.show()
