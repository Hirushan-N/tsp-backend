# app.py
import json

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
    return jsonify(
        {
            "bruteforce": "O(k!) where k is the number of selected cities (exact search over all permutations).",
            "nearest_neighbor": "O(k^2) - greedy algorithm: for each step, scan the remaining cities to find the nearest one.",
            "mst_prim": "O(k^2) - build a Minimum Spanning Tree with Prim's algorithm, then do a DFS traversal.",
            "random_search": "O(I * k) where I is the number of random permutations sampled.",
        }
    )


if __name__ == "__main__":
    import random 

    app.run(debug=True)
