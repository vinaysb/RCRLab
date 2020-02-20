# -*- coding: utf-8 -*-

"""Contains the code to carry-out reverse causal reasoning."""

import random
from typing import Optional, Iterator, List, Any, Tuple

import networkx as nx
import pandas as pd
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests


def do_rcr(
        kam_path: str,
        data: str = None,
        method: Optional[str] = 'fdr_bh'
) -> List[Tuple]:
    """
    Manages the RCR algorithm & its sub-functions.
    :param kam_path: Path to the file containing the source, relationship and the target nodes of a knowledge
    assembly model (KAM).
    :param data: Placeholder for the data that will be overlaid on the KAM.
    :param method: Method used for testing and adjustment of pvalues.
    :return: Concordance values for each hyp which are identified by their respective upstream node.
    """

    # Generate the knowledge assembly model
    kam = generate_kam(kam_path)

    # Overlay the data on the KAM
    # mixed_model = overlay_data(kam)
    mixed_model = overlay_random_data(kam)

    results = []

    # For all hyp's calculate the point probability
    for upstream, hyp in hyp_generator(mixed_model):

        if hyp:
            results.append((upstream, calculate_point_probability(mixed_model, hyp, upstream)))

    # Adjust the p values/probability with the help of family-wise error correction
    adjusted_pvals = list(multipletests([result[1] for result in results], method=method)[1])

    # Return the result as a list of tuple containing the upstream node and its adjusted p value
    return [(result[0], pval) for result, pval in zip(results, adjusted_pvals)]


def generate_kam(
        kam_path: str
) -> nx.DiGraph:
    """
    Generates the knowledge assembly model as a NetworkX graph.
    :param kam_path: Path to the file containing the source, relationship and the target nodes of a knowledge
    assembly model (KAM).
    :return: KAM graph as a NetworkX DiGraph.
    """

    # Read the file containing the kam file
    kam_df = pd.read_csv(kam_path, sep='\t', header=None)

    # Rename the column headers are Source, Relationship and Target
    kam_df.columns = ['Source', 'Relationship', 'Target']

    # Map relationships between the nodes as either +1 or -1 based on the interaction
    rlsp_mapping = {
        'activates': 1,
        'inhibits': -1
    }

    # Add the data to a directed graph
    kam = nx.DiGraph()

    for edge in kam_df.index:
        kam.add_edge(
            kam_df.at[edge, 'Source'],
            kam_df.at[edge, 'Target'],
            effect=rlsp_mapping[kam_df.at[edge, 'Relationship']]
        )

    return kam


def hyp_generator(
        mixed_model: nx.DiGraph
) -> Iterator[nx.DiGraph]:
    """
    Generates hyps based on the KAM.
    :param mixed_model: NetworkX DiGraph containing the KAM & the node attributes based on the data.
    :return: Sub-graph based on the depth first search of the graph for every node in the KAM.
    """

    for node in mixed_model.nodes:
        # Split the generated dfs into a list of lists based on the node
        yield (node, _list_split(list(nx.edge_dfs(mixed_model, node)), node))


def overlay_data(
        kam: nx.DiGraph,
        data: pd.DataFrame
) -> nx.DiGraph:
    """
    Overlays the data by assigning the nodes in the KAM with either '+1', '-1' or '0' based on the data.
    :param kam: NetworkX DiGraph containing the KAM.
    :param data: Data containing either up or down-regulation of the nodes in the KAM.
    :return: NetworkX DiGraph containing the KAM & the information from the data in the form of node attribute.
    """

    return NotImplemented


def overlay_random_data(
        kam: nx.DiGraph,
) -> nx.DiGraph:
    """
    Overlays the data by assigning the nodes in the KAM with either '+1', '-1' or '0' based on the data.
    :param kam: NetworkX DiGraph containing the KAM.
    :return: NetworkX DiGraph containing the KAM & the information from the data in the form of node attribute.
    """

    for node in kam.nodes:
        # For all chemicals the node attribute is ambiguous
        if node.startswith('CHEBI'):
            kam.nodes[node]["regulation"] = 0
        else:
            kam.nodes[node]["regulation"] = random.choice([1, -1, 0])

    return kam


def calculate_point_probability(
        graph: nx.DiGraph,
        hyp: List[List[Any]],
        upstream: Any
) -> float:
    """
    Calculate the probability of the number of successful predictions in a binomial distribution at random.
    :param graph: NetworkX DiGraph that contains the KAM & the data overlaid on the KAM.
    :param hyp: Current hypothesis whose probability need to be calculated.
    :param upstream: Upstream node of the hypothesis.
    :return: The probability.
    """

    # Keep a track of the number of positive and ambiguous nodes
    num_of_positives = 0
    num_of_ambiguous = 0

    # Loop over every branch in the hyp
    for branch in hyp:
        # Remember the attribute of the upstream node
        upstream_attr = graph.nodes[upstream]['regulation']

        # Initialize the effective attribute as the upstream node attribute
        effective_attr = upstream_attr

        for source, target in branch:
            # Check if the node is ambiguous
            if graph.nodes[target]['regulation'] == 0:
                num_of_ambiguous += 1

            # Multiply the node and the edge values to get the effective attribute value
            effective_attr *= graph.get_edge_data(source, target)['effect'] * graph.nodes[target]['regulation']

            # If after each node pass the effective attribute is equivalent to the upstream attribute then it is a
            # positive hit for that node.
            if effective_attr == upstream_attr:
                num_of_positives += 1

    # Flatten the hyp
    flat_hyp = set([node for branch in hyp for node in branch])

    # Calculate the pmf of the binomial distribution given the probability of achieving the result is 0.5
    return binom.pmf(num_of_positives, len(flat_hyp) - num_of_ambiguous, 0.5)


def _list_split(
        lst: List[Any],
        split_point: Any
) -> List[List[Any]]:
    """
    Splits a given lists into multiple lists based on the provided split points.
    :param lst: Given list that needs to be split.
    :param split_point: Element in the list that is used as a delimiter to split the list into different lists.
    :return: A list of lists containing the separated lists.
    """

    temp = []
    final = []

    for element in lst:
        if split_point in element:
            final.append(temp)
            temp = [element]
        else:
            temp.append(element)

    final.append(temp)
    final.remove([])

    return final
