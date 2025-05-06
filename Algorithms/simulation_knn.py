import pygame
import random
import math
from collections import defaultdict
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import csv
import os
import time
from datetime import datetime

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Traffic Management with KNN")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
LIGHT_GREEN = (100, 255, 100)

# Directions
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

# Intersection boundaries
INTERSECTION_TOP = 250
INTERSECTION_BOTTOM = 450
INTERSECTION_LEFT = 300
INTERSECTION_RIGHT = 500

# Efficiency tracker settings
EFFICIENCY_WINDOW_SIZE = 10  # Number of cycles to average for efficiency calculation
GRAPH_UPDATE_INTERVAL = 1000  # Update graph every 1000 milliseconds
CSV_SAVE_INTERVAL = 30000  # Save to CSV every 30 seconds

def create_intersection_background():
    background = pygame.Surface((WIDTH, HEIGHT))
    background.fill(GRAY)
    pygame.draw.rect(background, BLACK, (INTERSECTION_LEFT, 0, 200, HEIGHT))
    pygame.draw.rect(background, BLACK, (0, INTERSECTION_TOP, WIDTH, 200))
    for y in range(0, HEIGHT, 40):
        pygame.draw.rect(background, WHITE, (395, y, 10, 20))
    for x in range(0, WIDTH, 40):
        pygame.draw.rect(background, WHITE, (x, 345, 20, 10))
    return background

try:
    road_img = pygame.image.load('road.jpg')
    road_img = pygame.transform.scale(road_img, (WIDTH, HEIGHT))
except:
    road_img = create_intersection_background()

class Vehicle:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = random.uniform(2.0, 3.5)
        self.wait_time = 0
        self.has_entered_intersection = False
        self.has_exited_intersection = False
        self.entry_time = 0
        self.color = BLUE
        self.car_surface = self.create_car_surface()
        self.waiting = False
        self.total_wait_time = 0
        self.creation_time = pygame.time.get_ticks()

    def create_car_surface(self):
        car = pygame.Surface((40, 80), pygame.SRCALPHA)
        pygame.draw.rect(car, self.color, (0, 0, 40, 80))
        return car

    def update_color(self):
        if self.has_exited_intersection:
            self.color = LIGHT_GREEN
            self.car_surface = self.create_car_surface()
        elif self.waiting:
            self.color = RED
            self.car_surface = self.create_car_surface()
        else:
            self.color = BLUE
            self.car_surface = self.create_car_surface()

    def is_in_intersection(self):
        if self.direction in [NORTH, SOUTH]:
            return INTERSECTION_TOP < self.y < INTERSECTION_BOTTOM
        else:
            return INTERSECTION_LEFT < self.x < INTERSECTION_RIGHT

    def move(self, traffic_light_state, yellow_light):
        must_stop = False
        if traffic_light_state[self.direction] == RED or (yellow_light[self.direction] and not self.has_entered_intersection):
            must_stop = True
        
        if not must_stop:
            if self.waiting:
                self.waiting = False
                self.update_color()
                
            if self.direction == NORTH:
                self.y -= self.speed
                if not self.has_entered_intersection and self.y < INTERSECTION_BOTTOM:
                    self.has_entered_intersection = True
                    self.entry_time = pygame.time.get_ticks()
                if self.y < INTERSECTION_TOP:
                    self.has_exited_intersection = True
                    self.update_color()
            elif self.direction == EAST:
                self.x += self.speed
                if not self.has_entered_intersection and self.x > INTERSECTION_LEFT:
                    self.has_entered_intersection = True
                    self.entry_time = pygame.time.get_ticks()
                if self.x > INTERSECTION_RIGHT:
                    self.has_exited_intersection = True
                    self.update_color()
            elif self.direction == SOUTH:
                self.y += self.speed
                if not self.has_entered_intersection and self.y > INTERSECTION_TOP:
                    self.has_entered_intersection = True
                    self.entry_time = pygame.time.get_ticks()
                if self.y > INTERSECTION_BOTTOM:
                    self.has_exited_intersection = True
                    self.update_color()
            elif self.direction == WEST:
                self.x -= self.speed
                if not self.has_entered_intersection and self.x < INTERSECTION_RIGHT:
                    self.has_entered_intersection = True
                    self.entry_time = pygame.time.get_ticks()
                if self.x < INTERSECTION_LEFT:
                    self.has_exited_intersection = True
                    self.update_color()
            self.wait_time = 0
        else:
            self.wait_time += 1
            self.total_wait_time += 1
            if not self.waiting:
                self.waiting = True
                self.update_color()

    def draw(self):
        rotated_car = pygame.transform.rotate(self.car_surface, self.direction * -90)
        draw_x = self.x - rotated_car.get_width() // 2
        draw_y = self.y - rotated_car.get_height() // 2
        screen.blit(rotated_car, (draw_x, draw_y))

    def is_off_screen(self):
        buffer = 100
        if self.direction == NORTH and self.y < -buffer:
            return True
        elif self.direction == EAST and self.x > WIDTH + buffer:
            return True
        elif self.direction == SOUTH and self.y > HEIGHT + buffer:
            return True
        elif self.direction == WEST and self.x < -buffer:
            return True
        return False

    def get_journey_time(self):
        """Return the total time this vehicle has been in the simulation"""
        return pygame.time.get_ticks() - self.creation_time

class EfficiencyTracker:
    def __init__(self):
        self.data = []
        self.last_graph_update = pygame.time.get_ticks()
        self.last_csv_save = pygame.time.get_ticks()
        self.filename = f"traffic_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.header_written = False
        
        # Create plots folder if it doesn't exist
        os.makedirs("efficiency_data", exist_ok=True)
        
        # Initialize the figure for real-time plotting
        plt.ion()
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        self.fig.canvas.manager.set_window_title('Traffic Efficiency Metrics')

    def record_cycle(self, cycle_data):
        """Record data from a completed traffic light cycle"""
        self.data.append(cycle_data)
        
        # Keep only the most recent 100 cycles to prevent memory bloat
        if len(self.data) > 100:
            self.data = self.data[-100:]
        
        # Save data to CSV periodically
        current_time = pygame.time.get_ticks()
        if current_time - self.last_csv_save > CSV_SAVE_INTERVAL:
            self.save_to_csv()
            self.last_csv_save = current_time
            
        # Update plots periodically
        if current_time - self.last_graph_update > GRAPH_UPDATE_INTERVAL:
            self.update_plots()
            self.last_graph_update = current_time

    def calculate_average_efficiency(self):
        """Calculate average efficiency from recent cycles"""
        if not self.data:
            return 0
        
        recent_data = self.data[-EFFICIENCY_WINDOW_SIZE:] if len(self.data) >= EFFICIENCY_WINDOW_SIZE else self.data
        return sum(item['efficiency'] for item in recent_data) / len(recent_data)
    
    def calculate_throughput(self):
        """Calculate vehicles processed per minute"""
        if len(self.data) < 2:
            return 0
            
        recent_data = self.data[-EFFICIENCY_WINDOW_SIZE:] if len(self.data) >= EFFICIENCY_WINDOW_SIZE else self.data
        total_vehicles = sum(item['cleared'] for item in recent_data)
        time_span = (recent_data[-1]['time'] - recent_data[0]['time']) / 1000  # Convert to seconds
        return (total_vehicles / time_span) * 60 if time_span > 0 else 0  # Vehicles per minute
    
    def calculate_average_wait(self):
        """Calculate average wait time from recent cycles"""
        if not self.data:
            return 0
            
        recent_data = self.data[-EFFICIENCY_WINDOW_SIZE:] if len(self.data) >= EFFICIENCY_WINDOW_SIZE else self.data
        return sum(item['avg_wait'] for item in recent_data) / len(recent_data)
    
    def save_to_csv(self):
        """Save current efficiency data to CSV file"""
        try:
            with open(os.path.join("efficiency_data", self.filename), 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'cycle_number', 'cleared_vehicles', 'total_vehicles', 
                             'avg_wait', 'green_duration', 'efficiency', 'throughput']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not self.header_written:
                    writer.writeheader()
                    self.header_written = True
                
                for i, item in enumerate(self.data):
                    writer.writerow({
                        'timestamp': time.strftime('%H:%M:%S', time.localtime(item['time'] / 1000)),
                        'cycle_number': i + 1,
                        'cleared_vehicles': item['cleared'],
                        'total_vehicles': item['total_vehicles'],
                        'avg_wait': item['avg_wait'],
                        'green_duration': item['green_duration'] / 1000,  # Convert to seconds
                        'efficiency': item['efficiency'],
                        'throughput': item.get('throughput', 0)
                    })
            print(f"Data saved to {self.filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def update_plots(self):
        """Update the efficiency plots"""
        if len(self.data) < 2:
            return
            
        try:
            # Extract data for plotting
            times = [item['time'] for item in self.data]
            relative_times = [(t - times[0]) / 1000 / 60 for t in times]  # Minutes since start
            efficiencies = [item['efficiency'] for item in self.data]
            wait_times = [item['avg_wait'] for item in self.data]
            vehicles_cleared = [item['cleared'] for item in self.data]
            green_durations = [item['green_duration'] / 1000 for item in self.data]  # Convert to seconds
            
            # Calculate rolling averages for smoother plots
            window = 3 if len(self.data) >= 3 else 1
            rolling_efficiency = [sum(efficiencies[max(0, i-window+1):i+1])/min(window, i+1) for i in range(len(efficiencies))]
            rolling_wait = [sum(wait_times[max(0, i-window+1):i+1])/min(window, i+1) for i in range(len(wait_times))]
            
            # Clear previous plots
            for ax in self.axs.flatten():
                ax.clear()
                
            # Plot 1: Efficiency over time
            self.axs[0, 0].plot(relative_times, rolling_efficiency, 'g-', label='Rolling Avg')
            self.axs[0, 0].plot(relative_times, efficiencies, 'g.', alpha=0.3)
            self.axs[0, 0].set_title('Traffic Efficiency')
            self.axs[0, 0].set_xlabel('Time (minutes)')
            self.axs[0, 0].set_ylabel('Efficiency %')
            self.axs[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Average wait time
            self.axs[0, 1].plot(relative_times, rolling_wait, 'r-', label='Rolling Avg')
            self.axs[0, 1].plot(relative_times, wait_times, 'r.', alpha=0.3)
            self.axs[0, 1].set_title('Average Wait Time')
            self.axs[0, 1].set_xlabel('Time (minutes)')
            self.axs[0, 1].set_ylabel('Wait Frames')
            self.axs[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Vehicles cleared per cycle
            self.axs[1, 0].bar(relative_times, vehicles_cleared, width=0.02, color='blue', alpha=0.7)
            self.axs[1, 0].set_title('Vehicles Cleared Per Cycle')
            self.axs[1, 0].set_xlabel('Time (minutes)')
            self.axs[1, 0].set_ylabel('Count')
            self.axs[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Green light duration
            self.axs[1, 1].plot(relative_times, green_durations, 'y-')
            self.axs[1, 1].set_title('Green Light Duration')
            self.axs[1, 1].set_xlabel('Time (minutes)')
            self.axs[1, 1].set_ylabel('Seconds')
            self.axs[1, 1].grid(True, alpha=0.3)
            
            # Update the figure
            self.fig.tight_layout()
            plt.pause(0.01)
        except Exception as e:
            print(f"Error updating plots: {e}")

class TrafficDataCollector:
    def __init__(self):
        self.history = []
        self.current_state = {}
        self.features = []
        self.labels = []

    def record_state(self, vehicle_counts, wait_times, current_light):
        self.current_state = {
            'north_count': vehicle_counts[NORTH],
            'east_count': vehicle_counts[EAST],
            'south_count': vehicle_counts[SOUTH],
            'west_count': vehicle_counts[WEST],
            'north_wait': wait_times[NORTH],
            'east_wait': wait_times[EAST],
            'south_wait': wait_times[SOUTH],
            'west_wait': wait_times[WEST],
            'current_light': current_light
        }

    def record_outcome(self, vehicles_cleared, avg_wait_time, density_efficiency):
        self.current_state['outcome'] = vehicles_cleared * 10 - avg_wait_time + density_efficiency
        self.history.append(self.current_state)
        features = [
            self.current_state['north_count'],
            self.current_state['east_count'],
            self.current_state['south_count'],
            self.current_state['west_count'],
            self.current_state['north_wait'],
            self.current_state['east_wait'],
            self.current_state['south_wait'],
            self.current_state['west_wait'],
            self.current_state['current_light']
        ]
        self.features.append(features)
        self.labels.append(self.current_state['outcome'])

class KNNTrafficController:
    def __init__(self):
        self.model = KNeighborsRegressor(n_neighbors=3)
        self.is_trained = False
        self.imputer = SimpleImputer(strategy='mean')

    def train(self, features, labels):
        if len(features) > 10:
            X = np.array(features)
            y = np.array(labels)
            X = self.imputer.fit_transform(X)
            self.model.fit(X, y)
            self.is_trained = True
            print("KNN model trained with", len(features), "samples")

    def predict_best_duration(self, current_state):
        if not self.is_trained:
            # Default duration based on highest density
            max_direction = max([NORTH, EAST, SOUTH, WEST], 
                              key=lambda d: current_state[f"{['north', 'east', 'south', 'west'][d]}_count"])
            base_duration = 5000
            density_factor = current_state[f"{['north', 'east', 'south', 'west'][max_direction]}_count"] * 100
            return min(10000, max(3000, base_duration + density_factor))
            
        features = [
            current_state['north_count'],
            current_state['east_count'],
            current_state['south_count'],
            current_state['west_count'],
            current_state['north_wait'],
            current_state['east_wait'],
            current_state['south_wait'],
            current_state['west_wait'],
            current_state['current_light']
        ]
        features = self.imputer.transform([features])
        prediction = self.model.predict(features)[0]
        return max(3000, min(10000, 5000 + prediction * 100))

class TrafficLightSystem:
    def __init__(self):
        self.states = [RED] * 4
        self.yellow_states = [False] * 4
        self.current_green = None
        self.green_duration = 5000
        self.yellow_duration = 2000
        self.last_change_time = pygame.time.get_ticks()
        self.sequence = [NORTH, EAST, SOUTH, WEST]
        self.data_collector = TrafficDataCollector()
        self.knn_controller = KNNTrafficController()
        self.efficiency_tracker = EfficiencyTracker()
        self.avg_wait_before = 0
        self.last_cycle_vehicles_crossed = 0
        self.cycle_count = 0

    def check_intersection_clear(self, vehicles):
        for vehicle in vehicles:
            if vehicle.is_in_intersection() and vehicle.direction != self.current_green:
                return False
        return True

    def calculate_density_efficiency(self, vehicle_counts, current_green):
        """Calculate efficiency based on how well we're serving the highest density direction"""
        if current_green is None:
            return 0
            
        max_direction = max([NORTH, EAST, SOUTH, WEST], key=lambda d: vehicle_counts[d])
        max_count = vehicle_counts[max_direction]
        current_count = vehicle_counts[current_green]
        
        if max_count == 0:
            return 0
            
        # Efficiency is higher when we're giving green to the highest density direction
        if current_green == max_direction:
            return 100 * (current_count / max_count)
        else:
            return -50 * (1 - (current_count / max_count))

    def update(self, vehicles):
        current_time = pygame.time.get_ticks()
        time_since_change = current_time - self.last_change_time
        vehicle_counts = defaultdict(int)
        wait_times = defaultdict(int)
        
        # Calculate vehicle counts and wait times
        for vehicle in vehicles:
            vehicle_counts[vehicle.direction] += 1
            wait_times[vehicle.direction] += vehicle.wait_time
        
        for direction in wait_times:
            if vehicle_counts[direction] > 0:
                wait_times[direction] /= vehicle_counts[direction]
        
        self.avg_wait_before = sum(wait_times.values()) / 4 if len(wait_times) > 0 else 0
        self.data_collector.record_state(vehicle_counts, wait_times, self.current_green)

        # Calculate current efficiency metrics
        cleared = sum(1 for v in vehicles if v.has_exited_intersection and v.entry_time > self.last_change_time)
        avg_wait = sum(wait_times.values()) / 4 if len(wait_times) > 0 else 0
        
        # Calculate density-based efficiency
        density_efficiency = self.calculate_density_efficiency(vehicle_counts, self.current_green)
        
        # Combined efficiency metric (50% density efficiency, 30% wait time reduction, 20% throughput)
        efficiency_percentage = (0.5 * density_efficiency) +  0.3 * (100 * (1 - avg_wait/max(1, self.avg_wait_before))) +  0.2 * (cleared * 100 / max(1, sum(vehicle_counts.values())))

        # Calculate throughput (vehicles per minute)
        if self.cycle_count > 0:
            throughput = (cleared / (time_since_change / 1000)) * 60  # Vehicles per minute
        else:
            throughput = 0
            
        # Record cycle efficiency data
        cycle_data = {
            "time": current_time,
            "cleared": cleared,
            "total_vehicles": len(vehicles),
            "avg_wait": avg_wait,
            "green_duration": self.green_duration,
            "efficiency": efficiency_percentage,
            "throughput": throughput
        }
        
        # Check if it's time to change traffic light
        if time_since_change > self.green_duration + self.yellow_duration:
            # When cycle completes, record the outcome and efficiency data
            self.data_collector.record_outcome(cleared, avg_wait, density_efficiency)
            self.efficiency_tracker.record_cycle(cycle_data)
            self.cycle_count += 1
            
            # Train KNN model with collected data
            self.knn_controller.train(self.data_collector.features, self.data_collector.labels)
            
            # Switch to next light in sequence
            if self.current_green is None:
                self.current_green = self.sequence[0]
            else:
                current_index = self.sequence.index(self.current_green)
                self.current_green = self.sequence[(current_index + 1) % 4]
            
            # Get current traffic state for KNN prediction
            current_state = {
                'north_count': vehicle_counts[NORTH],
                'east_count': vehicle_counts[EAST],
                'south_count': vehicle_counts[SOUTH],
                'west_count': vehicle_counts[WEST],
                'north_wait': wait_times.get(NORTH, 0),
                'east_wait': wait_times.get(EAST, 0),
                'south_wait': wait_times.get(SOUTH, 0),
                'west_wait': wait_times.get(WEST, 0),
                'current_light': self.current_green
            }
            
            # Predict optimal green duration based on current state
            self.green_duration = self.knn_controller.predict_best_duration(current_state)
            self.last_change_time = current_time
            self.last_cycle_vehicles_crossed = cleared

        # Update traffic light states
        for i in range(4):
            if i == self.current_green:
                if time_since_change > self.green_duration:
                    self.states[i] = YELLOW
                    self.yellow_states[i] = True
                else:
                    self.states[i] = GREEN
                    self.yellow_states[i] = False
            else:
                self.states[i] = RED
                self.yellow_states[i] = False

    def draw(self):
        light_positions = [
            (WIDTH // 2 - 20, INTERSECTION_TOP - 70),
            (INTERSECTION_RIGHT + 70, HEIGHT // 2 - 20),
            (WIDTH // 2 + 20, INTERSECTION_BOTTOM + 70),
            (INTERSECTION_LEFT - 70, HEIGHT // 2 + 20)
        ]
        for i, pos in enumerate(light_positions):
            pygame.draw.rect(screen, BLACK, (pos[0] - 15, pos[1] - 45, 30, 90))
            pygame.draw.circle(screen, RED if self.states[i] == RED else (50, 0, 0), (pos[0], pos[1] - 30), 10)
            pygame.draw.circle(screen, YELLOW if self.states[i] == YELLOW else (50, 50, 0), (pos[0], pos[1]), 10)
            pygame.draw.circle(screen, GREEN if self.states[i] == GREEN else (0, 50, 0), (pos[0], pos[1] + 30), 10)

def main():
    running = True
    clock = pygame.time.Clock()
    vehicles = []
    spawn_timer = 0
    traffic_light_system = TrafficLightSystem()
    font = pygame.font.SysFont('Arial', 24)
    small_font = pygame.font.SysFont('Arial', 18)
    vehicle_counters = {NORTH: 0, EAST: 0, SOUTH: 0, WEST: 0}
    total_crossed = 0
    
    # Start time for simulation
    start_time = pygame.time.get_ticks()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Vehicle spawning logic
        spawn_timer += 1
        if spawn_timer >= 30:
            direction = random.randint(0, 3)
            if direction == NORTH:
                x = random.randint(INTERSECTION_LEFT + 50, INTERSECTION_RIGHT - 50)
                y = HEIGHT + 30
            elif direction == EAST:
                x = -30
                y = random.randint(INTERSECTION_TOP + 50, INTERSECTION_BOTTOM - 50)
            elif direction == SOUTH:
                x = random.randint(INTERSECTION_LEFT + 50, INTERSECTION_RIGHT - 50)
                y = -30
            else:
                x = WIDTH + 30
                y = random.randint(INTERSECTION_TOP + 50, INTERSECTION_BOTTOM - 50)
            vehicles.append(Vehicle(x, y, direction))
            vehicle_counters[direction] += 1
            spawn_timer = 0
            
        # Update traffic light system
        traffic_light_system.update(vehicles)
        
        # Update and remove vehicles
        vehicles_to_remove = []
        for vehicle in vehicles:
            vehicle.move(traffic_light_system.states, traffic_light_system.yellow_states)
            if vehicle.is_off_screen():
                vehicles_to_remove.append(vehicle)
                if vehicle.has_exited_intersection:
                    total_crossed += 1
        
        for vehicle in vehicles_to_remove:
            vehicles.remove(vehicle)
            
        # Draw everything
        screen.blit(road_img, (0, 0))
        
        # Sort vehicles to ensure proper drawing order
        vehicles_sorted = sorted(vehicles, key=lambda v: (-v.y if v.direction == NORTH else v.x if v.direction == EAST else v.y if v.direction == SOUTH else -v.x))
        for vehicle in vehicles_sorted:
            vehicle.draw()
            
        traffic_light_system.draw()
        
        # Calculate current traffic stats
        current_counts = {
            NORTH: len([v for v in vehicles if v.direction == NORTH]), 
            EAST: len([v for v in vehicles if v.direction == EAST]), 
            SOUTH: len([v for v in vehicles if v.direction == SOUTH]), 
            WEST: len([v for v in vehicles if v.direction == WEST])
        }
        
        wait_times = defaultdict(int)
        for vehicle in vehicles:
            wait_times[vehicle.direction] += vehicle.wait_time
            
        for direction in wait_times:
            if current_counts[direction] > 0:
                wait_times[direction] /= current_counts[direction]
                
        current_green = traffic_light_system.current_green if traffic_light_system.current_green is not None else -1
        green_text = ["NORTH", "EAST", "SOUTH", "WEST"][current_green] if current_green != -1 else "NONE"
        
        # Calculate average efficiency
        avg_efficiency = traffic_light_system.efficiency_tracker.calculate_average_efficiency()
        avg_wait = traffic_light_system.efficiency_tracker.calculate_average_wait()
        throughput = traffic_light_system.efficiency_tracker.calculate_throughput()
        
        # Running time in minutes and seconds
        runtime = (pygame.time.get_ticks() - start_time) / 1000  # seconds
        minutes = int(runtime // 60)
        seconds = int(runtime % 60)
        
        # Display stats
        stats = [
            f"Simulation Time: {minutes:02d}:{seconds:02d} | Cycle: {traffic_light_system.cycle_count}",
            f"Vehicles: {len(vehicles)} | Crossed: {total_crossed} | Last Cycle: {traffic_light_system.last_cycle_vehicles_crossed}",
            f"Green Light: {green_text} | Duration: {traffic_light_system.green_duration / 1000:.1f}s",
            f"North: {current_counts[NORTH]} (Wait: {wait_times.get(NORTH, 0):.1f})",
            f"East: {current_counts[EAST]} (Wait: {wait_times.get(EAST, 0):.1f})",
            f"South: {current_counts[SOUTH]} (Wait: {wait_times.get(SOUTH, 0):.1f})",
            f"West: {current_counts[WEST]} (Wait: {wait_times.get(WEST, 0):.1f})",
            f"KNN Model: {'Trained' if traffic_light_system.knn_controller.is_trained else 'Training...'}",
            f"Data Samples: {len(traffic_light_system.data_collector.features)}",
            f"Efficiency: {avg_efficiency:.2f}% | Avg Wait: {avg_wait:.1f}",
            f"Throughput: {throughput:.1f} vehicles/min"
        ]
        
        # Draw stats on screen
        for i, stat in enumerate(stats):
            text = font.render(stat, True, WHITE) if i < 3 else small_font.render(stat, True, WHITE)
            screen.blit(text, (10, 10 + i * 25))
            
        pygame.display.flip()
        clock.tick(60)
    
    # Before exiting, save final data
    traffic_light_system.efficiency_tracker.save_to_csv()
    plt.close('all')  # Close all matplotlib windows
    pygame.quit()

if __name__ == "__main__":
    main()