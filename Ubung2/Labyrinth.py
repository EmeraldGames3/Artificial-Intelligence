import heapq

import numpy as np


def distance(x, y):
    return (x[0] - y[0]) ** 2 + ((x[1] - y[1]) ** 2)


class Map:
    def __init__(self, m: np.ndarray) -> None:
        self.m = m

    def neighbors(self, cell):
        nrow, ncol = m.shape
        x, y = cell
        nb = []
        if x > 0:
            if m[x - 1, y] == 0:
                nb = nb + [(x - 1, y)]
        if x < (nrow - 1):
            if m[x + 1, y] == 0:
                nb = nb + [(x + 1, y)]
        if y > 0:
            if m[x, y - 1] == 0:
                nb = nb + [(x, y - 1)]
        if y < (ncol - 1):
            if m[x, y + 1] == 0:
                nb = nb + [(x, y + 1)]
        return nb


m = np.array(
    [[0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 1],
     [0, 1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 1, 1, 0, 0]])
mm = Map(m)
mm.neighbors((4, 1))


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class Stack:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.pop()


def make_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def dfs(map, start, goal):
    stack = Stack()
    stack.put(start)
    visited = set()
    came_from = {start: None}

    while not stack.empty():
        current = stack.get()
        if current == goal:
            return make_path(came_from, start, goal)

        for next in map.neighbors(current):
            if next not in visited:
                stack.put(next)
                visited.add(next)
                came_from[next] = current
    return None


def astar(map, start, goal):
    pass


# Testing dfs function from (0,0) to (5,6)
path_dfs = dfs(mm, (0, 0), (5, 6))

print(path_dfs)
# print(path_astar)
