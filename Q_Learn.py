import numpy as np
import matplotlib.pyplot as plt

# Define constants
GRID_SIZE = 5
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995

# Initialize variables for grid setup
grid = np.zeros((GRID_SIZE, GRID_SIZE))  # 0: empty, -1: obstacle, 1: start, 2: goal
start_pos = None
goal_pos = None
obstacle_positions = []

# Initialize visualization for grid setup
fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis', 3)
img = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=2, interpolation='nearest')
ax.set_title('Grid Setup (Left-click to toggle)')
plt.colorbar(img, ax=ax, ticks=[-1, 0, 1, 2], label='State')

# Function to handle mouse click events for grid setup
def onclick(event):
    global grid, start_pos, goal_pos, obstacle_positions
    
    row, col = int(event.ydata + 0.5), int(event.xdata + 0.5)  # Get clicked row and column
    
    if event.button == 1:  # Left-click: cycle through states (empty -> start -> goal -> obstacle)
        if (row, col) == start_pos:
            grid[row, col] = 0
            start_pos = None
        elif (row, col) == goal_pos:
            grid[row, col] = 0
            goal_pos = None
        elif (row, col) in obstacle_positions:
            grid[row, col] = 0
            obstacle_positions.remove((row, col))
        elif grid[row, col] == 0:
            if start_pos is None:
                grid[row, col] = 1  # Set as start
                start_pos = (row, col)
            elif goal_pos is None:
                grid[row, col] = 2  # Set as goal
                goal_pos = (row, col)
            else:
                grid[row, col] = -1  # Set as obstacle
                obstacle_positions.append((row, col))
    
    # Update visualization
    img.set_data(grid)
    fig.canvas.draw()

# Connect mouse click event to the plot
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Wait for grid setup to finish
print("Set up the grid (Left-click to toggle states). Close the plot window when done.")
plt.show()

# After grid setup, run Q-learning algorithm
if start_pos is None or goal_pos is None:
    print("Please set a start position and a goal position on the grid.")
else:
    # Initialize Q-table
    Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # 4 possible actions (up, down, left, right)
    
    # Define actions (up, down, left, right)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (row_change, col_change)
    
    # Helper function to perform epsilon-greedy action selection
    def epsilon_greedy_action(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)  # Choose random action
        else:
            return np.argmax(Q[state[0], state[1]])
    
    # Function to update visualization
    def update_visualization(Q, final_state=None):
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('viridis', 5)  # Define colormap
        optimal_actions = np.argmax(Q, axis=2)
        ax.imshow(optimal_actions, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
        ax.set_title('Q-Learning Visualization')
        
        # Highlight the final state in yellow
        if final_state is not None:
            ax.add_patch(plt.Rectangle((final_state[1] - 0.5, final_state[0] - 0.5), 1, 1, fill=False, edgecolor='yellow', linewidth=2))
        
        plt.colorbar(img, ax=ax, ticks=[0, 1, 2, 3, 4], label='Optimal Action')
        plt.show()
    
    # Q-learning algorithm
    goal_reached_count = 0
    epsilon = INITIAL_EPSILON
    final_state = start_pos
    for episode in range(NUM_EPISODES):
        state = start_pos  # Start at defined start position
        for step in range(MAX_STEPS_PER_EPISODE):
            action = epsilon_greedy_action(state, epsilon)
            next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
            
            # Check boundaries
            if next_state[0] < 0 or next_state[0] >= GRID_SIZE or next_state[1] < 0 or next_state[1] >= GRID_SIZE:
                reward = -10
                next_state = state  # Stay in the same state
            else:
                # Reward based on next state
                if grid[next_state[0], next_state[1]] == -1:  # Hit an obstacle
                    reward = -10
                    next_state = state  # Stay in the same state
                elif grid[next_state[0], next_state[1]] == 2:  # Reached the goal
                    reward = 100  # Higher reward for reaching the goal
                    goal_reached_count += 1
                    next_state = None  # Terminal state
                else:
                    reward = -1  # Slight penalty for each move
            
            # Q-value update using Bellman equation
            if next_state is not None:
                Q[state[0], state[1], action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])
            else:
                Q[state[0], state[1], action] += LEARNING_RATE * (reward - Q[state[0], state[1], action])
                break
            
            state = next_state
        
        # Decay epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        
        final_state = state  # Update final state
    
    # Final visualization after training
    update_visualization(Q, final_state)
    
    # Print the number of times the goal was reached
    print(f"The algorithm reached the goal {goal_reached_count} times.")
