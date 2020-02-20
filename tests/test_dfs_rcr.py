import unittest
import pandas as pd

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
        real_data = pd.read_csv(TEST_DATA_PATH, sep='\t', header=None)
        pred_data = dfs_rcr.RCR.generate_kam(TEST_DATA_PATH)

        real_nodes = set(list(real_data[0].values) + list(real_data[2].values))
        pred_nodes = set(pred_data.nodes)

        real_edges = set((i, j) for i, j in zip(real_data[0].values, real_data[2].values))
        pred_edges = set(pred_data.edges)

        self.assertSetEqual(pred_nodes, real_nodes)
        self.assertSetEqual(pred_edges, real_edges)

    def test_calculate_probability(self):
        mixed_model = pickle.load(open(TEST_MIXED_MODEL_PATH, 'rb'))
        upstream, hyp = pickle.load(open(TEST_HYP_PATH, 'rb'))

        self.assertEqual(
            dfs_rcr.RCR.calculate_probability(
                mixed_model,
                hyp,
                upstream
            ),
            0.25000000000000006
        )
