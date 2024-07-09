import numpy as np
import matplotlib.pyplot as plt

# Define constants
GRID_SIZE = 5
NUM_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95  # Increased discount factor
INITIAL_EPSILON = 1.0  # Start with high epsilon for exploration
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.995

# Define the grid environment with more obstacles
grid = np.array([
    [0, 0, 0, -1, 0],
    [0, -1, 0, -1, 0],
    [0, 0, 0, 0, 0],
    [-1, -1, 0, -1, 0],
    [0, 0, 0, 0, 1]
])

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

# Initialize visualization
fig, ax = plt.subplots()
cmap = plt.get_cmap('viridis', 5)  # Define colormap
img = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
ax.set_title('Q-Learning Visualization')
plt.colorbar(img, ax=ax, ticks=[0, 1, 2, 3, 4], label='Optimal Action')

# Function to update visualization
def update_visualization(Q, final_state=None):
    optimal_actions = np.argmax(Q, axis=2)
    img.set_data(optimal_actions)
    
    # Highlight the final state in yellow
    if final_state is not None:
        ax.add_patch(plt.Rectangle((final_state[1] - 0.5, final_state[0] - 0.5), 1, 1, fill=False, edgecolor='yellow', linewidth=2))
    
    return img,

# Q-learning algorithm with visualization
goal_reached_count = 0
final_state = (0, 0)
epsilon = INITIAL_EPSILON
for episode in range(NUM_EPISODES):
    state = (0, 0)  # Start at top-left corner
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
            elif grid[next_state[0], next_state[1]] == 1:  # Reached the goal
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
    
    final_state = state  # Update final state
    
    # Decay epsilon
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    
    # Update visualization every 100 episodes
    if episode % 100 == 0:
        update_visualization(Q)
        fig.canvas.draw()
        plt.pause(0.5)  # Pause to update plot (adjust the duration as needed)

# Final visualization after training with highlight
update_visualization(Q, final_state)
plt.show()

# Print the number of times the goal was reached
print(f"The algorithm reached the goal {goal_reached_count} times.")
