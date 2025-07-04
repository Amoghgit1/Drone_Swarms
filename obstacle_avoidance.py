import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class Boid:
    def __init__(self, position, velocity, max_speed=0.1, max_force=0.01):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.max_speed = max_speed
        self.max_force = max_force
        self.perception = 0.5  # Radius for neighbor detection
    
    def update(self, boids):
        # Calculate three main boid rules
        alignment = self.align(boids)
        cohesion = self.cohere(boids)
        separation = self.separate(boids)
        
        # Apply weights to each rule (can be adjusted)
        self.velocity += alignment * 1.0
        self.velocity += cohesion * 1.0
        self.velocity += separation * 1.5
        
        # Limit velocity
        norm = np.linalg.norm(self.velocity)
        if norm > self.max_speed:
            self.velocity = self.velocity / norm * self.max_speed
        
        # Update position
        self.position += self.velocity
        
        # Boundary conditions (wrap around)
        self.position = np.mod(self.position, 1.0)
    
    def align(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_velocity = np.zeros(2)
        
        for boid in boids:
            if np.linalg.norm(boid.position - self.position) < self.perception:
                avg_velocity += boid.velocity
                total += 1
        
        if total > 0:
            avg_velocity /= total
            avg_velocity = (avg_velocity / np.linalg.norm(avg_velocity)) * self.max_speed
            steering = avg_velocity - self.velocity
            if np.linalg.norm(steering) > self.max_force:
                steering = (steering / np.linalg.norm(steering)) * self.max_force
        
        return steering
    
    def cohere(self, boids):
        steering = np.zeros(2)
        total = 0
        center_of_mass = np.zeros(2)
        
        for boid in boids:
            if np.linalg.norm(boid.position - self.position) < self.perception:
                center_of_mass += boid.position
                total += 1
        
        if total > 0:
            center_of_mass /= total
            vec_to_com = center_of_mass - self.position
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) * self.max_speed
            steering = vec_to_com - self.velocity
            if np.linalg.norm(steering) > self.max_force:
                steering = (steering / np.linalg.norm(steering)) * self.max_force
        
        return steering
    
    def separate(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)
        
        for boid in boids:
            distance = np.linalg.norm(boid.position - self.position)
            if 0 < distance < self.perception:
                diff = self.position - boid.position
                diff /= distance  # Weight by distance
                avg_vector += diff
                total += 1
        
        if total > 0:
            avg_vector /= total
            if np.linalg.norm(avg_vector) > 0:
                avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * self.max_speed
            steering = avg_vector - self.velocity
            if np.linalg.norm(steering) > self.max_force:
                steering = (steering / np.linalg.norm(steering)) * self.max_force
        
        return steering

def create_drone_patch(position, angle):
    """Create a triangular patch representing a drone at given position and angle"""
    size = 0.02
    points = np.array([
        [size, 0],
        [-size/2, size/2],
        [-size/2, -size/2]
    ])
    
    # Rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    points = np.dot(points, rot)
    
    # Translation
    points += position
    return Polygon(points, closed=True, color='blue')

def update(frame, boids, collection, ax):
    """Update function for animation"""
    patches = []
    for boid in boids:
        boid.update(boids)
        angle = np.arctan2(boid.velocity[1], boid.velocity[0])
        patches.append(create_drone_patch(boid.position, angle))
    
    # Update the collection
    collection.set_paths(patches)
    
    # Add some random obstacles (optional)
    if frame % 50 == 0 and frame > 0:
        ax.plot(np.random.rand(), np.random.rand(), 'ro', markersize=10)
    
    return collection,

def main():
    # Set up the matplotlib backend to be interactive
    plt.switch_backend('TkAgg')  # or 'Qt5Agg' depending on your system
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_facecolor('lightgray')
    ax.set_title('Swarm Drone Simulation using Boids Algorithm')
    
    # Create boids (drones)
    num_boids = 8
    boids = [Boid(
        position=np.random.rand(2),
        velocity=(np.random.rand(2) - 0.5) * 0.05
        ) for _ in range(num_boids)]
    
    # Create initial patches
    patches = [create_drone_patch(b.position, 
              np.arctan2(b.velocity[1], b.velocity[0])) for b in boids]
    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=60, fargs=(boids, collection, ax),
        interval=100, blit=False)
    
    # To save the animation instead of showing (uncomment if needed)
    # ani.save('swarm_drones.mp4', writer='ffmpeg', fps=30)
    
    plt.show()

if __name__ == "__main__":
    main()
