import unittest
import networkx as nx
import networkx.algorithms.isomorphism as iso

import dfs_rcr

import os
import pickle

HERE = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(HERE, 'test_data.tsv')
TEST_GRAPH_PATH = os.path.join(HERE, 'test_graph.pkl')
TEST_MIXED_MODEL_PATH = os.path.join(HERE, 'test_mixed_model.pkl')
TEST_HYP_PATH = os.path.join(HERE, 'test_hyp.pkl')


class TestDfsRcr(unittest.TestCase):

    def test_generate_kam(self):
        graph = pickle.load(open(TEST_GRAPH_PATH, 'rb'))

        em = iso.numerical_edge_match('Relationship', 1)

        self.assertTrue(nx.is_isomorphic(dfs_rcr.RCR.generate_kam(TEST_DATA_PATH), graph, em))

    def test_calculate_probability(self):
        mixed_model = pickle.load(open(TEST_MIXED_MODEL_PATH, 'rb'))
        upstream, hyp = pickle.load(open(TEST_HYP_PATH, 'rb'))

        self.assertEqual(
            dfs_rcr.RCR.calculate_probability(
                mixed_model,
                hyp,
                upstream
            ),
            0.15624999999999994
        )
