"""
Week 3 - Uninformed Search: Breadth-First and Depth-First Search (Lab VII, AI & ML)
"""

from collections import deque

# ----------------------------------------------------------------------
# Q1 - Maze solving with BFS: shortest path from start to goal
# ----------------------------------------------------------------------
# 0 = free, 1 = wall. Start (0, 0), goal (4, 4).
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
]


def bfs_shortest_path(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


path = bfs_shortest_path(maze, (0, 0), (4, 4))
print("BFS shortest path through the maze:", path)
print("Steps:", len(path) - 1)

# ----------------------------------------------------------------------
# Q2 - Family tree search with DFS: all descendants of a person
# ----------------------------------------------------------------------
family_tree = {
    "Aarav": {"Rohan": {"Kabir": {}, "Anaya": {}},
              "Meera": {"Arjun": {}}}
}


def dfs_descendants(tree, person):
    descendants = []
    children = tree.get(person, {})
    for child in children:
        descendants.append(child)
        descendants.extend(dfs_descendants(children, child))
    return descendants


# Flatten the nested tree so any person can be looked up by name
def flatten(tree, flat=None):
    flat = {} if flat is None else flat
    for person, children in tree.items():
        flat[person] = children
        flatten(children, flat)
    return flat


flat_tree = flatten(family_tree)
print("Descendants of Rohan:", dfs_descendants(flat_tree, "Rohan"))
print("Descendants of Aarav:", dfs_descendants(flat_tree, "Aarav"))

# ----------------------------------------------------------------------
# Q3 - Robot in a grid: BFS, moving only right or down
# ----------------------------------------------------------------------
# 0 = free, 1 = blocked. Start (0, 0), goal (3, 3).
grid = [
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
]


def bfs_right_down(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in [(0, 1), (1, 0)]:  # right, down only
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


robot_path = bfs_right_down(grid, (0, 0), (3, 3))
print("Robot path (right/down only):", robot_path)

# ----------------------------------------------------------------------
# Q4 - Website navigation tree: DFS path to a given page
# ----------------------------------------------------------------------
site = {"Home": {"About": {},
                 "Products": {"Electronics": {}, "Books": {}},
                 "Contact": {}}}


def dfs_path(tree, target, path=None):
    path = [] if path is None else path
    for page, children in tree.items():
        if page == target:
            return path + [page]
        found = dfs_path(children, target, path + [page])
        if found:
            return found
    return None


print("Path to 'Books':", " -> ".join(dfs_path(site, "Books")))
