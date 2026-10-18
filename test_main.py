import unittest
import networkx as nx
from main import calculate

class TestGraphCentralities(unittest.TestCase):
    def test_path_middle_has_highest_betweenness(self):
        graph = nx.path_graph(["A", "B", "C"])
        rows = {row["vk_id"]: row for row in calculate(graph, {"A": "A", "B": "B", "C": "C"})}
        self.assertAlmostEqual(rows["B"]["betweenness"], 1.0)
        self.assertAlmostEqual(rows["A"]["betweenness"], 0.0)
        self.assertGreater(rows["B"]["closeness"], rows["A"]["closeness"])
        self.assertGreater(rows["B"]["eigenvector"], rows["A"]["eigenvector"])

    def test_isolated_member_is_retained(self):
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B"])
        rows = calculate(graph, {"A": "A", "B": "B"})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["eigenvector"] == 0.0 for row in rows))

if __name__ == "__main__":
    unittest.main()
