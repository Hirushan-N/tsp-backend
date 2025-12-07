import json
import os
import unittest

from app import (
    generate_random_matrix,
    tsp_bruteforce,
    tsp_nearest_neighbor,
    tsp_random_search,
    route_distance,
)


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.matrix = generate_random_matrix(5)
        self.home = 0
        self.selected = [1, 2, 3, 4]

    def test_matrix_symmetry(self):
        for i in range(5):
            for j in range(5):
                if i == j:
                    self.assertEqual(self.matrix[i][j], 0)
                else:
                    self.assertEqual(self.matrix[i][j], self.matrix[j][i])

    def test_bruteforce_optimal(self):
        res = tsp_bruteforce(self.home, self.selected, self.matrix)
        self.assertIsNotNone(res["route"])
        self.assertTrue(len(res["route"]) == len(self.selected) + 2)

    def test_nearest_and_random_not_better_than_bruteforce(self):
        brute = tsp_bruteforce(self.home, self.selected, self.matrix)
        greedy = tsp_nearest_neighbor(self.home, self.selected, self.matrix)
        rnd = tsp_random_search(self.home, self.selected, self.matrix, iterations=200)

        self.assertLessEqual(brute["distance"], greedy["distance"])
        self.assertLessEqual(brute["distance"], rnd["distance"])


if __name__ == "__main__":
    unittest.main()
