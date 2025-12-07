import unittest

from app import (
    generate_random_matrix,
    tsp_bruteforce,
    tsp_nearest_neighbor,
    tsp_random_search,
    tsp_mst_prim,
    route_distance,
)


class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        # Use a small size for fast exact testing
        self.matrix = generate_random_matrix(5)
        self.home = 0
        self.selected = [1, 2, 3, 4]

    # -------------------------------
    # MATRIX TESTS
    # -------------------------------

    def test_matrix_symmetry(self):
        """Distance matrix must be symmetric with 0 on diagonal"""
        for i in range(5):
            for j in range(5):
                if i == j:
                    self.assertEqual(self.matrix[i][j], 0)
                else:
                    self.assertEqual(self.matrix[i][j], self.matrix[j][i])

    # -------------------------------
    # ROUTE DISTANCE TEST
    # -------------------------------

    def test_route_distance_calculation(self):
        route = [0, 1, 2, 3, 4, 0]
        dist = route_distance(self.matrix, route)
        self.assertIsInstance(dist, int)
        self.assertGreater(dist, 0)

    # -------------------------------
    # BRUTE FORCE TEST
    # -------------------------------

    def test_bruteforce_optimal(self):
        """Bruteforce must always return a valid optimal route"""
        res = tsp_bruteforce(self.home, self.selected, self.matrix)

        self.assertIsNotNone(res["route"])
        self.assertEqual(len(res["route"]), len(self.selected) + 2)
        self.assertEqual(res["route"][0], self.home)
        self.assertEqual(res["route"][-1], self.home)

    # -------------------------------
    # GREEDY + RANDOM + MST TESTS
    # -------------------------------

    def test_nearest_neighbor_valid(self):
        greedy = tsp_nearest_neighbor(self.home, self.selected, self.matrix)

        self.assertEqual(greedy["route"][0], self.home)
        self.assertEqual(greedy["route"][-1], self.home)
        self.assertEqual(len(greedy["route"]), len(self.selected) + 2)

    def test_random_search_valid(self):
        rnd = tsp_random_search(self.home, self.selected, self.matrix, iterations=200)

        self.assertEqual(rnd["route"][0], self.home)
        self.assertEqual(rnd["route"][-1], self.home)
        self.assertEqual(len(rnd["route"]), len(self.selected) + 2)

    def test_mst_prim_valid(self):
        mst = tsp_mst_prim(self.home, self.selected, self.matrix)

        self.assertEqual(mst["route"][0], self.home)
        self.assertEqual(mst["route"][-1], self.home)
        self.assertEqual(len(mst["route"]), len(self.selected) + 2)

    # -------------------------------
    # OPTIMALITY COMPARISON TEST
    # -------------------------------

    def test_heuristics_not_better_than_bruteforce(self):
        brute = tsp_bruteforce(self.home, self.selected, self.matrix)
        greedy = tsp_nearest_neighbor(self.home, self.selected, self.matrix)
        rnd = tsp_random_search(self.home, self.selected, self.matrix, iterations=200)
        mst = tsp_mst_prim(self.home, self.selected, self.matrix)

        self.assertLessEqual(brute["distance"], greedy["distance"])
        self.assertLessEqual(brute["distance"], rnd["distance"])
        self.assertLessEqual(brute["distance"], mst["distance"])


if __name__ == "__main__":
    unittest.main()
