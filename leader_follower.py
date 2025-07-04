import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Set backend
matplotlib.use('Agg')

class Boid:
    def __init__(self, position, velocity, max_speed=0.1, max_force=0.01, is_leader=False):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.max_speed = max_speed
        self.max_force = max_force
        self.perception = 0.5
        self.is_leader = is_leader
        self.path_index = 0
        self.leader_path = self.generate_random_path(8) if is_leader else []
        self.path_history = [position.copy()] if is_leader else []

    def generate_random_path(self, num_points):
        """Generate smooth random path using cubic spline interpolation"""
        # Generate random control points
        points = np.random.rand(num_points, 2)
        
        # Add some smoothing
        for i in range(1, num_points-1):
            points[i] = (points[i-1] + points[i] + points[i+1]) / 3
            
        return points.tolist()

    def update_leader_path(self):
        """Dynamically update path with new random points"""
        if self.is_leader and np.random.rand() < 0.03:  # 3% chance per frame to add new point
            # Generate new point in direction of current velocity
            new_dir = self.velocity / (np.linalg.norm(self.velocity) + 0.0001)
            new_point = self.position + new_dir * 0.3 + (np.random.rand(2)-0.5)*0.2
            new_point = np.clip(new_point, 0.1, 0.9)  # Keep within bounds
            
            self.leader_path.append(new_point.tolist())
            if len(self.leader_path) > 12:  # Limit path length
                self.leader_path.pop(0)
                self.path_index = max(0, self.path_index-1)

    def update(self, boids):
        if self.is_leader:
            self.update_leader_path()
            self.path_history.append(self.position.copy())
            if len(self.path_history) > 100:  # Limit history
                self.path_history.pop(0)

            if len(self.leader_path) > 0:
                target = np.array(self.leader_path[self.path_index])
                direction = target - self.position
                distance = np.linalg.norm(direction)
                
                if distance < 0.05:  # Reached target
                    self.path_index = (self.path_index + 1) % len(self.leader_path)
                    target = np.array(self.leader_path[self.path_index])
                    direction = target - self.position
                
                # Add some randomness to movement
                if distance > 0:
                    noise = (np.random.rand(2)-0.5)*0.02
                    self.velocity = (direction/distance + noise) * self.max_speed
                
                self.position += self.velocity
        else:
            # Follower boid behavior
            alignment = self.align(boids)
            cohesion = self.cohere(boids)
            separation = self.separate(boids)
            leader_attraction = self.attract_to_leader(boids)
            
            self.velocity += alignment * 1.0
            self.velocity += cohesion * 1.0
            self.velocity += separation * 1.5
            self.velocity += leader_attraction * 2
            
            speed = np.linalg.norm(self.velocity)
            if speed > self.max_speed:
                self.velocity = self.velocity / speed * self.max_speed
                
            self.position += self.velocity
        
        # Wrap around edges
        self.position = np.mod(self.position, 1.0)

    def attract_to_leader(self, boids):
        steering = np.zeros(2)
        for boid in boids:
            if boid.is_leader:
                vec = boid.position - self.position
                dist = np.linalg.norm(vec)
                if dist > 0:
                    vec = (vec / dist) * self.max_speed
                    steering = vec - self.velocity
                    if np.linalg.norm(steering) > self.max_force:
                        steering = steering / np.linalg.norm(steering) * self.max_force
                break
        return steering

    def align(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_velocity = np.zeros(2)
        
        for boid in boids:
            if not boid.is_leader and np.linalg.norm(boid.position - self.position) < self.perception:
                avg_velocity += boid.velocity
                total += 1
        
        if total > 0:
            avg_velocity /= total
            steering = (avg_velocity / np.linalg.norm(avg_velocity)) * self.max_speed - self.velocity
            if np.linalg.norm(steering) > self.max_force:
                steering = steering / np.linalg.norm(steering) * self.max_force
        
        return steering

    def cohere(self, boids):
        steering = np.zeros(2)
        total = 0
        center = np.zeros(2)
        
        for boid in boids:
            if not boid.is_leader and np.linalg.norm(boid.position - self.position) < self.perception:
                center += boid.position
                total += 1
        
        if total > 0:
            center /= total
            vec = center - self.position
            dist = np.linalg.norm(vec)
            if dist > 0:
                steering = (vec / dist) * self.max_speed - self.velocity
                if np.linalg.norm(steering) > self.max_force:
                    steering = steering / np.linalg.norm(steering) * self.max_force
        
        return steering

    def separate(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)
        
        for boid in boids:
            dist = np.linalg.norm(boid.position - self.position)
            if 0 < dist < self.perception * 0.5:  # Stronger at closer distances
                diff = self.position - boid.position
                diff /= dist
                avg_vector += diff
                total += 1
        
        if total > 0:
            avg_vector /= total
            steering = avg_vector * self.max_speed - self.velocity
            if np.linalg.norm(steering) > self.max_force:
                steering = steering / np.linalg.norm(steering) * self.max_force
        
        return steering

def create_drone_patch(position, angle, is_leader=False):
    size = 0.02
    points = np.array([
        [size, 0],
        [-size/2, size/2],
        [-size/2, -size/2]
    ])
    
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    points = np.dot(points, rot) + position
    
    color = 'red' if is_leader else 'blue'
    alpha = 1.0 if is_leader else 0.7
    return Polygon(points, closed=True, color=color, alpha=alpha)

def update(frame, boids, collection, ax):
    # Clear previous path drawings
    while len(ax.lines) > 0:
        ax.lines[0].remove()
    
    patches = []
    for boid in boids:
        boid.update(boids)
        angle = np.arctan2(boid.velocity[1], boid.velocity[0])
        patches.append(create_drone_patch(boid.position, angle, boid.is_leader))
    
    collection.set_paths(patches)
    
    # Draw leader path
    leader = next(b for b in boids if b.is_leader)
    if len(leader.leader_path) > 1:
        path = np.array(leader.leader_path)
        ax.plot(path[:,0], path[:,1], 'r-', alpha=0.3, linewidth=1)
    
    # Draw path history
    if len(leader.path_history) > 1:
        history = np.array(leader.path_history)
        ax.plot(history[:,0], history[:,1], 'r:', alpha=0.2, linewidth=0.5)
    
    return collection,

def main():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_facecolor('lightgray')
    ax.set_title('Swarm Drone Simulation with Randomized Leader Path')
    
    num_boids = 3
    boids = []
    
    # Create leader
    leader = Boid(
        position=[0.5, 0.5],
        velocity=[0.01, 0.01],
        is_leader=True,
        max_speed=0.030,
        max_force=0.02
    )
    boids.append(leader)
    
    # Create followers
    for _ in range(num_boids - 1):
        boids.append(Boid(
            position=np.random.rand(2),
            velocity=(np.random.rand(2)-0.5)*0.03,
            max_speed=0.12,
            max_force=0.015
        ))
    
    # Create initial patches
    patches = [create_drone_patch(b.position, 
              np.arctan2(b.velocity[1], b.velocity[0]), b.is_leader) for b in boids]
    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=100, fargs=(boids, collection, ax),
        interval=50, blit=False)
    
    # Save animation
    ani.save('swarm_random_leader.mp4', writer='ffmpeg', fps=16, dpi=100)
    print("Animation saved as 'swarm_random_leader.mp4'")

if __name__ == "__main__":
    main()
