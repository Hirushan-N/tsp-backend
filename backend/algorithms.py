# algorithms.py
import random
import time
from itertools import permutations
from typing import List, Dict, Any


def generate_random_matrix(n: int = 10, low: int = 50, high: int = 100) -> List[List[int]]:
    matrix: List[List[int]] = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            elif j < i:
                row.append(matrix[j][i])
            else:
                row.append(random.randint(low, high))
        matrix.append(row)
    return matrix


def route_distance(matrix: List[List[int]], route: List[int]) -> int:
    dist = 0
    for i in range(len(route) - 1):
        dist += matrix[route[i]][route[i + 1]]
    return dist


def tsp_bruteforce(home: int, selected: List[int], matrix: List[List[int]]) -> Dict[str, Any]:
    best_route = None
    best_distance = float("inf")

    for perm in permutations(selected):
        route = [home] + list(perm) + [home]
        d = route_distance(matrix, route)
        if d < best_distance:
            best_distance = d
            best_route = route

    return {"route": best_route, "distance": int(best_distance)}


def tsp_nearest_neighbor(home: int, selected: List[int], matrix: List[List[int]]) -> Dict[str, Any]:
    unvisited = set(selected)
    route = [home]
    current = home

    while unvisited:
        next_city = min(unvisited, key=lambda c: matrix[current][c])
        unvisited.remove(next_city)
        route.append(next_city)
        current = next_city

    route.append(home)
    d = route_distance(matrix, route)
    return {"route": route, "distance": int(d)}


def tsp_random_search(
    home: int,
    selected: List[int],
    matrix: List[List[int]],
    iterations: int = 2000,
) -> Dict[str, Any]:
    k = len(selected)

    if k <= 7:
        iters = list(permutations(selected))
    else:
        iters = []
        for _ in range(iterations):
            perm = random.sample(selected, k)
            iters.append(perm)

    best_route = None
    best_distance = float("inf")
    for perm in iters:
        route = [home] + list(perm) + [home]
        d = route_distance(matrix, route)
        if d < best_distance:
            best_distance = d
            best_route = route

    return {"route": best_route, "distance": int(best_distance)}


def tsp_mst_prim(home: int, selected: List[int], matrix: List[List[int]]) -> Dict[str, Any]:
    subset = set(selected)
    subset.add(home)

    visited = {home}
    edges_mst = [] 

    while visited != subset:
        best_edge = None
        best_weight = float("inf")

        for u in visited:
            for v in subset - visited:
                w = matrix[u][v]
                if w < best_weight:
                    best_weight = w
                    best_edge = (u, v)

        if best_edge is None:
            break

        u, v = best_edge
        edges_mst.append((u, v))
        visited.add(v)

    adj: Dict[int, List[int]] = {}
    for u, v in edges_mst:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    route_order: List[int] = []

    def dfs(node: int, parent: int = -1):
        route_order.append(node)
        for nxt in adj.get(node, []):
            if nxt != parent:
                dfs(nxt, node)

    dfs(home)

    route = route_order + [home]
    d = route_distance(matrix, route)
    return {"route": route, "distance": int(d)}


def run_algorithms(home: int, selected: List[int], matrix: List[List[int]]) -> Dict[str, Dict[str, Any]]:

    algorithms = {
        "bruteforce": tsp_bruteforce,    
        "nearest_neighbor": tsp_nearest_neighbor, 
        "mst_prim": tsp_mst_prim,
        "random_search": tsp_random_search, 
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, fn in algorithms.items():
        start = time.perf_counter()
        result = fn(home, selected, matrix)
        duration_ms = (time.perf_counter() - start) * 1000.0
        result["durationMs"] = duration_ms
        results[name] = result

    return results
