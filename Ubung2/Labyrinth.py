import matplotlib.pyplot as plt
import numpy as np

from DataStructeres import Stack, PriorityQueue


def distance(x, y):
    return (x[0] - y[0]) ** 2 + ((x[1] - y[1]) ** 2)


class Map:
    def __init__(self, m: np.ndarray) -> None:
        self.m = m

    @staticmethod
    def neighbors(cell):
        now, col = m.shape
        x, y = cell
        nb = []
        if x > 0:
            if m[x - 1, y] == 0:
                nb = nb + [(x - 1, y)]
        if x < (now - 1):
            if m[x + 1, y] == 0:
                nb = nb + [(x + 1, y)]
        if y > 0:
            if m[x, y - 1] == 0:
                nb = nb + [(x, y - 1)]
        if y < (col - 1):
            if m[x, y + 1] == 0:
                nb = nb + [(x, y + 1)]
        return nb

    @staticmethod
    def cost(node1, node2):
        return 1

    @staticmethod
    def heuristic(node, goal_node):
        return distance(node, goal_node)


m = np.array(
    [[0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 1],
     [0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 1, 1, 0, 0]])
mm = Map(m)


# mm.neighbors((4, 1))

# for l in m:
#     print(l)
# print()


def make_path(came_from, start_node, goal_node):
    current = goal_node
    path = []
    while current != start_node:
        path.append(current)
        current = came_from[current]
    path.append(start_node)
    path.reverse()
    return path


def dfs(labyrinth, start_node, goal_node):
    stack = Stack()
    stack.put(start_node)
    visited = set()
    came_from = {start_node: None}

    while not stack.empty():
        current = stack.get()
        if current == goal_node:
            return make_path(came_from, start_node, goal_node)

        for next_node in labyrinth.neighbors(current):
            if next_node not in visited:
                stack.put(next_node)
                visited.add(next_node)
                came_from[next_node] = current
    return None


def astar(labyrinth, start_node, goal_node):
    queue = PriorityQueue()
    queue.put(start_node, labyrinth.heuristic(start_node, goal_node))
    # Used for reconstructing the path at the end
    came_from = {start_node: None}
    # Cost to travel to each node
    cost = {start_node: 0}

    # Loop until there are no nodes left to explore
    while not queue.empty():
        # Get the current node with the lowest cost
        current = queue.get()

        # If the goal has been reached, reconstruct and return the path
        if current == goal_node:
            return make_path(came_from, start_node, goal_node)

        # Explore the neighbors of the current node
        for neighbor in labyrinth.neighbors(current):
            # Calculate the new cost to the neighbor
            new_cost = cost[current] + labyrinth.cost(current, neighbor)

            if neighbor not in cost or new_cost < cost[neighbor]:
                # Update the cost and path to the neighbor
                cost[neighbor] = new_cost
                priority = new_cost + labyrinth.heuristic(neighbor, goal_node)
                queue.put(neighbor, priority)
                came_from[neighbor] = current

    # If the goal was never reached
    return None


# Run DFS and A_Star to find the path
start = (0, 0)
goal = (5, 6)
path_dfs = dfs(mm, start, goal)
path_astar = astar(mm, start, goal)


# Visualize the map and path
def visualize_map_and_path(map_data, path):
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # Set up the colormap: obstacles are yellow, path and free space are purple
    norm = plt.Normalize(vmin=map_data.min(), vmax=map_data.max())

    # Create a color mapping: 1 is mapped to yellow, 0 is purple
    colors = [[norm(0), "purple"], [norm(1), "yellow"]]
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)

    # Plot the map
    ax.imshow(map_data, cmap=cmap, norm=norm)

    # Plot the path
    if path:
        # Convert a path to a numpy array for convenience
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], color='black', linewidth=2)

    # Turn off the axis labels
    ax.axis('off')

    # Show the plot
    plt.show()


# Call the visualization function
visualize_map_and_path(m, path_dfs)
visualize_map_and_path(m, path_astar)

# print(path_dfs)
# print(path_astar)
