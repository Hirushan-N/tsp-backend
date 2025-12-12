import json
import random

from flask import Flask, request, jsonify
from flask_cors import CORS

from config import CITIES
from db import get_db, init_db
from algorithms import (
    generate_random_matrix,
    route_distance,
    run_algorithms,
)

app = Flask(__name__)
CORS(app)

init_db()



def _parse_round_payload(data):
    player_name = (data.get("playerName") or "").strip()
    route_between = data.get("routeBetween") or []
    home_city = data.get("homeCity")
    matrix = data.get("distanceMatrix")

    if not player_name:
        return None, jsonify({"error": "playerName is required"}), 400
    if not route_between:
        return None, jsonify({"error": "routeBetween must contain at least one city"}), 400
    if not home_city:
        return None, jsonify({"error": "homeCity is required"}), 400
    if home_city not in CITIES:
        return None, jsonify({"error": "Invalid homeCity"}), 400
    if not isinstance(matrix, list) or not matrix:
        return None, jsonify({"error": "distanceMatrix is required"}), 400

    n = len(CITIES)
    if len(matrix) != n or any(not isinstance(row, list) or len(row) != n for row in matrix):
        return None, jsonify({"error": f"distanceMatrix must be a {n}x{n} matrix"}), 400
    for i in range(n):
        if matrix[i][i] != 0:
            return None, jsonify({"error": "distanceMatrix diagonal must be 0"}), 400

    try:
        selected_indices = [CITIES.index(c) for c in route_between]
    except ValueError:
        return None, jsonify({"error": "Unknown city in routeBetween"}), 400

    home_index = CITIES.index(home_city)

    if home_city in route_between:
        return None, jsonify({"error": "routeBetween must not include home city"}), 400

    if len(set(selected_indices)) != len(selected_indices):
        return None, jsonify({"error": "routeBetween must not contain duplicate cities"}), 400

    if len(selected_indices) > 9:
        return None, jsonify({"error": "Please choose at most 8 cities to keep the game fast."}), 400

    return {
        "player_name": player_name,
        "route_between": route_between,
        "home_city": home_city,
        "home_index": home_index,
        "matrix": matrix,
        "selected_indices": selected_indices,
    }, None, None

@app.route("/api/new-game", methods=["POST"])
def new_game():
    try:
        matrix = generate_random_matrix(len(CITIES))
        home_index = random.randint(0, len(CITIES) - 1)
        home_city = CITIES[home_index]

        return jsonify(
            {
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

    parsed, err_resp, err_code = _parse_round_payload(data)
    if err_resp is not None:
        return err_resp, err_code

    try:
        player_name = parsed["player_name"]
        route_between = parsed["route_between"]
        home_city = parsed["home_city"]
        home_index = parsed["home_index"]
        matrix = parsed["matrix"]
        selected_indices = parsed["selected_indices"]

        algo_results = run_algorithms(home_index, selected_indices, matrix)

        optimal = algo_results["bruteforce"]
        optimal_route = optimal["route"]
        optimal_distance = optimal["distance"]

        user_route_indices = [home_index] + [CITIES.index(c) for c in route_between] + [home_index]
        user_distance = route_distance(matrix, user_route_indices)

        correct = user_route_indices == optimal_route

        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO sessions (home_city, distance_matrix) VALUES (?, ?)",
            (home_city, json.dumps(matrix)),
        )
        session_id = cur.lastrowid

        cur.execute("INSERT OR IGNORE INTO players (name) VALUES (?)", (player_name,))
        cur.execute("SELECT id FROM players WHERE name = ?", (player_name,))
        player_id = cur.fetchone()["id"]

        for name, res in algo_results.items():
            cur.execute(
                """
                INSERT INTO algorithm_runs (session_id, algorithm_name, duration_ms, distance)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, name, float(res["durationMs"]), int(res["distance"])),
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
            "sessionId": session_id,
            "correct": correct,
            "homeCity": home_city,
            "yourRoute": [CITIES[i] for i in user_route_indices],
            "yourDistance": user_distance,
            "optimalRoute": [CITIES[i] for i in optimal_route],
            "optimalDistance": optimal_distance,
            "algorithms": {
                name: {
                    "route": [CITIES[i] for i in res["route"]],
                    "distance": int(res["distance"]),
                    "durationMs": float(res["durationMs"]),
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
    return jsonify(
        {
            "bruteforce": "O(k!) where k is the number of selected cities (exact search over all permutations).",
            "nearest_neighbor": "O(k^2) - greedy algorithm: for each step, scan the remaining cities to find the nearest one.",
            "mst_prim": "O(k^2) - build a Minimum Spanning Tree with Prim's algorithm, then do a DFS traversal.",
            "random_search": "O(I * k) where I is the number of random permutations sampled.",
        }
    )


@app.route("/api/performance", methods=["GET"])
def performance():
    try:
        limit = request.args.get("limit", default=15, type=int)
        if limit <= 0 or limit > 200:
            limit = 15

        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                s.id AS session_id,
                s.home_city,
                s.created_at,
                p.name AS player_name
            FROM sessions s
            LEFT JOIN games g ON g.session_id = s.id
            LEFT JOIN players p ON p.id = g.player_id
            ORDER BY s.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        sessions = cur.fetchall()

        if not sessions:
            conn.close()
            return jsonify({"rounds": []})

        session_ids = [s["session_id"] for s in sessions]

        placeholders = ",".join(["?"] * len(session_ids))
        cur.execute(
            f"""
            SELECT session_id, algorithm_name, duration_ms
            FROM algorithm_runs
            WHERE session_id IN ({placeholders})
            """,
            session_ids,
        )
        runs = cur.fetchall()
        conn.close()

        runs_map = {}
        for r in runs:
            sid = r["session_id"]
            runs_map.setdefault(sid, {})
            runs_map[sid][r["algorithm_name"]] = float(r["duration_ms"])

        rounds = []
        for s in reversed(sessions):
            sid = s["session_id"]
            algo = runs_map.get(sid, {})

            rounds.append(
                {
                    "sessionId": sid,
                    "playerName": s["player_name"] if s["player_name"] else None,
                    "homeCity": s["home_city"],
                    "createdAt": s["created_at"],
                    "bruteforce": algo.get("bruteforce"),
                    "nearest_neighbor": algo.get("nearest_neighbor"),
                    "mst_prim": algo.get("mst_prim"),
                    "random_search": algo.get("random_search"),
                }
            )

        return jsonify({"rounds": rounds})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
