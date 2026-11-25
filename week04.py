"""
Week 4 - Informed Search: Best-First Search and A* (Lab VII, AI & ML)
"""

import heapq
import math
from collections import deque

# ----------------------------------------------------------------------
# Q1 - Grid pathfinding with A*: Manhattan distance heuristic
# ----------------------------------------------------------------------
# 0 = walkable, 1 = blocked. Start (0, 0), goal (4, 4).
grid = [
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_list = [(manhattan(start, goal), 0, start, [start])]
    best_g = {start: 0}
    visited = 0
    while open_list:
        f, g, (r, c), path = heapq.heappop(open_list)
        visited += 1
        if (r, c) == goal:
            return path, visited
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0):
                new_g = g + 1
                if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                    best_g[(nr, nc)] = new_g
                    heapq.heappush(open_list, (
                        new_g + manhattan((nr, nc), goal),
                        new_g, (nr, nc), path + [(nr, nc)]))
    return None, visited


astar_path, astar_visited = a_star(grid, (0, 0), (4, 4))
print("A* path:", astar_path)
print("A* steps:", len(astar_path) - 1, "| nodes visited:", astar_visited)

# ----------------------------------------------------------------------
# Q2 - City-to-city travel with Best-First Search (Euclidean heuristic)
# ----------------------------------------------------------------------
cities = {"A": (0, 0), "B": (2, 3), "C": (5, 2), "D": (6, 6), "E": (8, 3)}

# Roads between cities (cost = straight-line distance).
roads = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B", "D", "E"],
    "D": ["B", "C", "E"],
    "E": ["C", "D"],
}


def euclidean(a, b):
    return math.dist(cities[a], cities[b])


def best_first_search(start, goal):
    # Greedy choice: always expand the unvisited city closest to the goal
    # (straight-line distance), ignoring the distance already travelled.
    open_list = [(euclidean(start, goal), start, [start])]
    visited = {start}
    while open_list:
        h, city, path = heapq.heappop(open_list)
        if city == goal:
            return path
        for nxt in roads[city]:
            if nxt not in visited:
                visited.add(nxt)
                heapq.heappush(open_list,
                               (euclidean(nxt, goal), nxt, path + [nxt]))
    return None


print("Best-First Search path A -> E:", best_first_search("A", "E"))

# ----------------------------------------------------------------------
# Q3 - Maze solving: A* vs BFS (steps taken, nodes visited)
# ----------------------------------------------------------------------
# S = start (0, 0), G = goal (0, 4), 1 = wall, 0 = free.
maze = [
    ["S", 0, 1, 0, "G"],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
maze_grid = [[0 if cell in ("S", "G", 0) else 1 for cell in row]
             for row in maze]


def bfs(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path, len(visited)
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None, len(visited)


bfs_path, bfs_visited = bfs(maze_grid, (0, 0), (0, 4))
astar_path3, astar_visited3 = a_star(maze_grid, (0, 0), (0, 4))
print("Maze BFS  -> steps:", len(bfs_path) - 1,
      "| nodes visited:", bfs_visited, "| path:", bfs_path)
print("Maze A*  -> steps:", len(astar_path3) - 1,
      "| nodes visited:", astar_visited3, "| path:", astar_path3)

# ----------------------------------------------------------------------
# Q4 - Treasure hunt on a grid: A* with cliffs and bonus cells
# ----------------------------------------------------------------------
# Cost per cell: 2 = normal ground, 1 = bonus cell, None = cliff (blocked).
# Start S (0, 0), treasure T (4, 4).
treasure_grid = [
    [2, 2, None, 2, 2],
    [2, 1, 2, 2, 2],
    [2, 2, 1, None, 2],
    [None, 2, 2, 1, 2],
    [2, 2, 2, 2, 2],
]


def a_star_weighted(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    # Manhattan heuristic is admissible: cheapest step costs 1.
    open_list = [(manhattan(start, goal), 0, start, [start])]
    best_g = {start: 0}
    while open_list:
        f, g, (r, c), path = heapq.heappop(open_list)
        if (r, c) == goal:
            return path, g
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cost = grid[nr][nc]
                if cost is None:
                    continue
                new_g = g + cost
                if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                    best_g[(nr, nc)] = new_g
                    heapq.heappush(open_list, (
                        new_g + manhattan((nr, nc), goal),
                        new_g, (nr, nc), path + [(nr, nc)]))
    return None, None


treasure_path, treasure_cost = a_star_weighted(
    treasure_grid, (0, 0), (4, 4))
print("Treasure path:", treasure_path)
print("Total cost:", treasure_cost)
