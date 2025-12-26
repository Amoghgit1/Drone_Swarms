import cv2
import numpy as np
import time
#from picamera2 import Picamera2
import matplotlib.pyplot as plt
from collections import deque

class MineDetector:
    def __init__(self):
        #start camera
        global cap
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

         # Training data storage
        self.mine_features = []
        self.red_marker_features = []
        
        # Detection parameters
        self.mine_threshold = 0.7
        self.red_threshold = 0.6
        
        # Grid for path planning (10x10 grid)
        self.grid_size = (10, 10)
        self.mine_positions = set()
        self.current_pos = (0, 0)
        self.goal_pos = None
    
    def extract_features(self, image):
        """Extract color histogram and edge features from image"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Combine features
        features = np.concatenate([hist_h, hist_s, hist_v])
        return features
    
    def train_mine_detector(self, mine_images_path):
        """Train on mine images"""
        print("Training mine detector...")
        import os
        
        for img_file in os.listdir(mine_images_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg','.webp')):
                img_path = os.path.join(mine_images_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    features = self.extract_features(img)
                    self.mine_features.append(features)
                    print(f"Trained on mine image: {img_file}")
        
        print(f"Mine training complete! Loaded {len(self.mine_features)} samples")

    def train_red_marker_detector(self, red_marker_images_path):
        """Train on red marker images"""
        print("Training red marker detector...")
        import os
        
        for img_file in os.listdir(red_marker_images_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg','.webp')):
                img_path = os.path.join(red_marker_images_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    features = self.extract_features(img)
                    self.red_marker_features.append(features)
                    print(f"Trained on red marker: {img_file}")
        
        print(f"Red marker training complete! Loaded {len(self.red_marker_features)} samples")

    def compare_features(self, feat1, feat2):
        """Compare two feature vectors using correlation"""
        return cv2.compareHist(feat1.reshape(-1, 1), feat2.reshape(-1, 1), cv2.HISTCMP_CORREL)
    
    def detect_mine(self, image):
        """Detect if image contains a mine"""
        if not self.mine_features:
            return False, 0.0
        
        features = self.extract_features(image)
        max_similarity = 0.0
        
        for mine_feat in self.mine_features:
            similarity = self.compare_features(features, mine_feat)
            max_similarity = max(max_similarity, similarity)
        
        is_mine = max_similarity > self.mine_threshold
        return is_mine, max_similarity
    
    def detect_red_marker(self, image):
        """Detect if image contains red marker (termination signal)"""
        if not self.red_marker_features:
            # Fallback: detect red color directly
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            red_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            return red_ratio > 0.3, red_ratio
            #print("Red ratio:", red_ratio)

        
        features = self.extract_features(image)
        max_similarity = 0.0
        
        for red_feat in self.red_marker_features:
            similarity = self.compare_features(features, red_feat)
            max_similarity = max(max_similarity, similarity)
        
        is_red = max_similarity > self.red_threshold
        return is_red, max_similarity
    
    def update_grid_position(self, grid_x, grid_y):
        """Update current position on grid"""
        self.current_pos = (grid_x, grid_y)
    
    def add_mine_to_grid(self, grid_x, grid_y):
        """Mark a mine location on the grid"""
        self.mine_positions.add((grid_x, grid_y))
    
    def a_star_pathfinding(self, start, goal):
        """A* pathfinding algorithm to find safe path"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    if (nx, ny) not in self.mine_positions:
                        neighbors.append((nx, ny))
            return neighbors
        
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            open_set.remove(current)
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    open_set.add(neighbor)
        
        return None  # No path found
    
    def visualize_grid_and_path(self, path=None):
        """Visualize the grid with mines and safe path"""
        grid = np.ones((self.grid_size[0], self.grid_size[1], 3))
        
        # Mark mines in red
        for mine_pos in self.mine_positions:
            grid[mine_pos[0], mine_pos[1]] = [1, 0, 0]
        
        # Mark current position in blue
        grid[self.current_pos[0], self.current_pos[1]] = [0, 0, 1]
        
        # Mark goal in green
        if self.goal_pos:
            grid[self.goal_pos[0], self.goal_pos[1]] = [0, 1, 0]
        
        # Mark path in yellow
        if path:
            for pos in path:
                if pos != self.current_pos and pos != self.goal_pos:
                    grid[pos[0], pos[1]] = [1, 1, 0]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, interpolation='nearest')
        plt.grid(True, which='both', color='black', linewidth=0.5)
        plt.xticks(range(self.grid_size[1]))
        plt.yticks(range(self.grid_size[0]))
        plt.title('Mine Field Map & Safe Path')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Mines'),
            Patch(facecolor='blue', label='Current Position'),
            Patch(facecolor='green', label='Goal'),
            Patch(facecolor='yellow', label='Safe Path')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.savefig('path_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Path visualization saved as 'path_visualization.png'")
    
    def run_detection(self, duration=60):
        """Run live detection from camera"""
        print("\n=== Starting Mine Detection System ===")
        print("Press 'q' to quit manually\n")
        
        #self.picam.start()
        time.sleep(2)  # Camera warm-up
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Capture frame
                #frame = self.picam.capture_array()
                frame_available, frame = cap.read()
                if not frame_available:
                    print("Failed to capture frame from camera.")
                    break
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_count += 1
                #cv2.imwrite(f'frame_{frame_count}.jpg', frame)
                
                # Check for red marker (termination condition)
                is_red, red_conf = self.detect_red_marker(frame)
                if is_red:
                    print(f"\n[TERMINATION] Red marker detected! (confidence: {red_conf:.2f})")
                    cv2.imwrite(f'red_marker_detected.jpg', frame)
                    print("Saved: red_marker_detected.jpg")
                    
                    # Set goal position
                    self.goal_pos = self.current_pos
                    break
                
                # Check for mines
                is_mine, mine_conf = self.detect_mine(frame)
                if is_mine:
                    print(f"[MINE DETECTED] Frame {frame_count} - Confidence: {mine_conf:.2f}")
                    cv2.imwrite(f'mine_detected_{frame_count}.jpg', frame)
                    print(f"Saved: mine_detected_{frame_count}.jpg")
                    
                    # Add to grid (simulate grid position)
                    grid_x = frame_count % self.grid_size[0]
                    grid_y = frame_count // self.grid_size[0] % self.grid_size[1]
                    self.add_mine_to_grid(grid_x, grid_y)
                    print(f"Mine marked at grid position: ({grid_x}, {grid_y})")
                
                # Update position simulation
                if frame_count % 5 == 0:
                    grid_x = frame_count // 5 % self.grid_size[0]
                    grid_y = frame_count // 10 % self.grid_size[1]
                    self.update_grid_position(grid_x, grid_y)
                
                # Display frame
                display_frame = frame.copy()
                status_text = "RED MARKER!" if is_red else ("MINE!" if is_mine else "Clear")
                color = (0, 0, 255) if is_red else ((0, 165, 255) if is_mine else (0, 255, 0))
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow('Drone Mine Detection', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.5)  # Simulate drone movement
        
        finally:
            #self.picam.stop()
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"\n=== Detection Complete ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Mines detected: {len(self.mine_positions)}")
        print(f"Mine positions: {self.mine_positions}")
        
        # Path planning
        if self.goal_pos and self.mine_positions:
            print("\n=== Computing Safe Path ===")
            start = (0, 0)
            path = self.a_star_pathfinding(start, self.goal_pos)
            
            if path:
                print(f"Safe path found! Length: {len(path)} steps")
                print(f"Path: {' -> '.join([str(p) for p in path])}")
                self.visualize_grid_and_path(path)
            else:
                print("No safe path found! All routes blocked by mines.")
                self.visualize_grid_and_path()
        else:
            print("\n=== Final Grid State ===")
            self.visualize_grid_and_path()


# Example usage
if __name__ == "__main__":
    detector = MineDetector()
    
    # Train the detector (replace with your actual paths)
    detector.train_mine_detector('./mine_training_images/')
    detector.train_red_marker_detector('./red_training_image/')
    
    #print("NOTE: Place training images in folders and uncomment training lines above")
    print("Starting detection in 3 seconds...")
    time.sleep(3)
    
    # Run detection for 60 seconds or until red marker found
    detector.run_detection(duration=60)
