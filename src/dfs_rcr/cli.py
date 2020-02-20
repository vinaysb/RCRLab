# -*- coding: utf-8 -*-

"""A command-line interface for depth first search based reverse causal reasoning."""

import logging

import click

from dfs_rcr.RCR import do_rcr

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--kam-path',
    help="Path to the Knowledge assembly model as a TSV file",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
@click.option(
    '--data-path',
    help="Path to the gene expression data as a TSV file",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
@click.option(
    '--method',
    help="Method used for testing and adjustment of P-Values",
    type=str,
    required=False,
    default='fdr_bh',
    show_default=True,
)
def cli(
        kam_path: str,
        data_path: str,
        method: str,
):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    do_rcr(kam_path, data_path, method)

