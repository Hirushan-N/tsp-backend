import json
import random
import sqlite3
import time
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

DB_NAME = "tsp_game.db"
CITIES = [chr(ord("A") + i) for i in range(10)]

app = Flask(__name__)
CORS(app)



def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home_city TEXT NOT NULL,
            distance_matrix TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            session_id INTEGER NOT NULL,
            home_city TEXT NOT NULL,
            selected_cities TEXT NOT NULL,
            shortest_route TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(player_id) REFERENCES players(id),
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS algorithm_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            algorithm_name TEXT NOT NULL,
            duration_ms REAL NOT NULL,
            distance INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        );
        """
    )

    conn.commit()
    conn.close()


init_db()


def generate_random_matrix(n: int = 10, low: int = 50, high: int = 100) -> List[List[int]]:
    """
    Generate a symmetric distance matrix with 0 on the diagonal.
    """
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
    """
    Exact algorithm: tries all permutations of selected cities.
    Time complexity: O(k!) where k = number of selected cities.
    """
    from itertools import permutations

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
    """
    Greedy nearest neighbor algorithm.
    Time complexity: O(k^2)
    """
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
    """
    Randomized search over random permutations.
    Time complexity: O(iterations * k)
    """
    from itertools import permutations

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
    """
    MST-based TSP heuristic using Prim's algorithm (from the course's
    'Minimum Connector / Prim's Algorithm' content).

    Steps:
      1. Build a Minimum Spanning Tree (MST) over the subgraph containing
         the home city and all selected cities using Prim's algorithm.
      2. Perform a DFS/preorder traversal of the MST starting from home.
      3. Use the DFS order as the visiting order; return to home at the end.

    Time complexity:
      - Prim's algorithm here: O(k^2) for k = |selected| + 1
      - DFS traversal: O(k)
      => Overall: O(k^2)
    """
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


# ---------- API endpoints ----------

@app.route("/api/new-game", methods=["POST"])
def new_game():
    try:
        matrix = generate_random_matrix(len(CITIES))
        home_index = random.randint(0, len(CITIES) - 1)
        home_city = CITIES[home_index]

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (home_city, distance_matrix) VALUES (?, ?)",
            (home_city, json.dumps(matrix)),
        )
        session_id = cur.lastrowid
        conn.commit()
        conn.close()

        return jsonify(
            {
                "sessionId": session_id,
                "cities": CITIES,
                "homeCity": home_city,
                "homeIndex": home_index,
                "distanceMatrix": matrix,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/check-answer", methods=["POST"])
def check_answer():
    data = request.get_json(force=True, silent=True) or {}
    try:
        session_id = data.get("sessionId")
        player_name = (data.get("playerName") or "").strip()
        route_between = data.get("routeBetween") or []

        if not session_id:
            return jsonify({"error": "sessionId is required"}), 400
        if not player_name:
            return jsonify({"error": "playerName is required"}), 400
        if not route_between:
            return jsonify({"error": "routeBetween must contain at least one city"}), 400

        # Load session
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "Session not found"}), 404

        home_city = row["home_city"]
        matrix = json.loads(row["distance_matrix"])

        if home_city not in CITIES:
            conn.close()
            return jsonify({"error": "Invalid home city in DB"}), 500

        home_index = CITIES.index(home_city)

        try:
            selected_indices = list({CITIES.index(c) for c in route_between})
        except ValueError:
            conn.close()
            return jsonify({"error": "Unknown city in routeBetween"}), 400

        if home_index in selected_indices:
            conn.close()
            return jsonify({"error": "routeBetween must not include home city"}), 400

        if len(selected_indices) > 9:
            conn.close()
            return jsonify({"error": "Please choose at most 8 cities to keep the game fast."}), 400

        algo_results = run_algorithms(home_index, selected_indices, matrix)

        optimal = algo_results["bruteforce"]
        optimal_route = optimal["route"]
        optimal_distance = optimal["distance"]

        try:
            user_route_indices = [home_index] + [CITIES.index(c) for c in route_between] + [home_index]
        except ValueError:
            conn.close()
            return jsonify({"error": "Unknown city in routeBetween"}), 400

        user_distance = route_distance(matrix, user_route_indices)

        correct = user_route_indices == optimal_route

        cur.execute("INSERT OR IGNORE INTO players (name) VALUES (?)", (player_name,))
        cur.execute("SELECT id FROM players WHERE name = ?", (player_name,))
        player_id = cur.fetchone()["id"]

        for name, res in algo_results.items():
            cur.execute(
                """
                INSERT INTO algorithm_runs (session_id, algorithm_name, duration_ms, distance)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, name, res["durationMs"], res["distance"]),
            )

        if correct:
            selected_letters = ",".join(route_between)
            optimal_letters = ",".join(CITIES[i] for i in optimal_route)
            cur.execute(
                """
                INSERT INTO games (player_id, session_id, home_city, selected_cities, shortest_route)
                VALUES (?, ?, ?, ?, ?)
                """,
                (player_id, session_id, home_city, selected_letters, optimal_letters),
            )

        conn.commit()
        conn.close()

        response = {
            "correct": correct,
            "homeCity": home_city,
            "yourRoute": [CITIES[i] for i in user_route_indices],
            "yourDistance": user_distance,
            "optimalRoute": [CITIES[i] for i in optimal_route],
            "optimalDistance": optimal_distance,
            "algorithms": {
                name: {
                    "route": [CITIES[i] for i in res["route"]],
                    "distance": res["distance"],
                    "durationMs": res["durationMs"],
                }
                for name, res in algo_results.items()
            },
            "message": "Correct! Well done." if correct else "Not quite. Check the optimal route below.",
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/complexity", methods=["GET"])
def complexity():
    """
    Simple endpoint returning textual complexity analysis.
    """
    return jsonify(
        {
            "bruteforce": "O(k!) where k is the number of selected cities (exact search over all permutations).",
            "nearest_neighbor": "O(k^2) - greedy algorithm: for each step, scan the remaining cities to find the nearest one.",
            "mst_prim": "O(k^2) - build a Minimum Spanning Tree with Prim's algorithm, then do a DFS traversal.",
            "random_search": "O(I * k) where I is the number of random permutations sampled.",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
