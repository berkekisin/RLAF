#!/usr/bin/env python3
"""
generate_graph.py

Command-line tool that:
  1. Takes an Answer Set Program (ASP) as input (string or file).
  2. Reifies the program using clingox.reify.
  3. Combines the reified facts with a chosen meta-program.
  4. Calls clingo to compute (one) answer set.
  5. Extracts node/1, edge/2, and node_label/2 facts from the answer set.
  6. Writes:
       - adjacency.csv: adjacency matrix using integer node IDs
                        (edges are treated as directed)
       - nodes.csv:      mapping from integer ID to node label

Labeling semantics:
  - In the meta-program, you can define node_label(Node, Label).
  - If node_label/2 is present for a node, Label is used in nodes.csv.
  - Otherwise, the label is just the integer ID of the node.

Meta-program selection:
  - You can either:
      * provide your own meta-program via --meta-program-file, OR
      * choose one of the built-in meta-programs via --meta-program-label.

Nodes and edges are represented as strings internally.
"""

import argparse
import csv
import sys
from typing import Dict, Iterable, List, Set, Tuple

from clingo import Control, Model
from clingox import reify


# ---------------------------------------------------------------------------
# Built-in meta-programs
# ---------------------------------------------------------------------------

BUILTIN_META_PROGRAMS: Dict[str, str] = {

    "dependency": r"""
        atom(|L|) :- weighted_literal_tuple(_,L,_).
        atom(|L|) :- literal_tuple(_,L).
        atom(A)   :- atom_tuple(_,A).
        node(A)   :- atom(A).

        atom_in_body(A,Body) :-
            rule(_,Body), Body = normal(B),
            literal_tuple(B,L), A = |L|.
        atom_in_body(A,Body) :-
            rule(_,Body), Body = sum(B,_),
            weighted_literal_tuple(B,L,_), A = |L|.
        atom_in_head(A,Head) :-
            rule(Head,_), Head = disjunction(H),
            atom_tuple(H,A).
        atom_in_head(A,Head) :-
            rule(Head,_), Head = choice(H),
            atom_tuple(H,A).

        edge(A1,A2) :-
            rule(Head,Body),
            atom_in_head(A1,Head),
            atom_in_body(A2,Body).
        edge(A2,A1) :- edge(A1,A2).

        node_label(A,Label) :-
            output(Label,B), literal_tuple(B,A).

        #show node/1.
        #show edge/2.
        #show node_label/2.
    """,

}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reify an ASP program, combine with a meta-program, solve, and "
            "export a node/edge graph as an adjacency matrix CSV plus a node "
            "mapping CSV. Edges are treated as directed; enforce symmetry in "
            "the meta program if you want an undirected graph."
        )
    )

    # Exactly one of --program-file or --program (string)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--program-file",
        "-f",
        help="Path to a file containing the input ASP program.",
    )
    group.add_argument(
        "--program",
        "-p",
        help="ASP program given directly as a string.",
    )

    # Either a custom meta-program file OR a built-in label.
    meta_group = parser.add_mutually_exclusive_group()
    meta_group.add_argument(
        "--meta-program-file",
        "-m",
        help=(
            "Path to a file containing the meta-program that will be combined "
            "with the reified program."
        ),
    )
    meta_group.add_argument(
        "--meta-program-label",
        "-l",
        choices=sorted(BUILTIN_META_PROGRAMS.keys()),
        default="dependency",
        help=(
            "Label of a built-in meta-program to use. "
            "Ignored if --meta-program-file is given. "
            f"Available: {', '.join(sorted(BUILTIN_META_PROGRAMS.keys()))}. "
            "Default: dependency."
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        default="adjacency.csv",
        help="Path to the output adjacency matrix CSV file (default: adjacency.csv).",
    )

    parser.add_argument(
        "--nodes-output",
        "-n",
        default="nodes.csv",
        help="Path to the output node mapping CSV file (default: nodes.csv).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Program loading and reification
# ---------------------------------------------------------------------------

def load_program_source(args: argparse.Namespace) -> str:
    """
    Load the input ASP program as a string, either from a file or directly
    from the command line.
    """
    if args.program_file:
        with open(args.program_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return args.program


def load_meta_program_source(args: argparse.Namespace) -> str:
    """
    Load the meta-program that will be combined with the reified program.

    Precedence:
      - If --meta-program-file is given, use that file.
      - Otherwise, use the built-in meta-program selected by
        --meta-program-label (default: dependency).
    """
    if args.meta_program_file:
        with open(args.meta_program_file, "r", encoding="utf-8") as f:
            return f.read()

    label = args.meta_program_label or "dependency"
    return BUILTIN_META_PROGRAMS[label]


def reify_program(program_source: str) -> Iterable[str]:
    """
    Reify the given ASP program string using clingox.reify.

    The `reify.reify_program` function returns an iterable of symbols
    representing reified program facts.
    """
    return reify.reify_program(program_source)


# ---------------------------------------------------------------------------
# Solving and graph extraction
# ---------------------------------------------------------------------------

def combine_and_solve(
    reified_program: str,
    meta_program: str,
) -> Tuple[List[str], Set[Tuple[str, str]], Dict[str, str]]:
    """
    Create a clingo.Control object, add the reified program and meta-program,
    solve for a single answer set, and extract node/1, edge/2, node_label/2.

    Returns:
        ordered_nodes: list of node identifiers (strings) in the order
                       they first appear in the model
        edges:         set of directed edges (u, v)
        label_map:     mapping node_id_string -> label_string

    If you want an undirected graph, enforce symmetry in the meta program
    (e.g., edge(B,A) :- edge(A,B).).
    """
    ctl = Control()

    # Add reified program facts (as a single string)
    ctl.add("base", [], reified_program)

    # Add the meta-program
    ctl.add("base", [], meta_program)

    # Ground the combined program
    ctl.ground([("base", [])])

    ordered_nodes: List[str] = []
    seen_nodes: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()
    label_map: Dict[str, str] = {}

    # Solve and only consider the first model (answer set)
    with ctl.solve(yield_=True) as handle:
        for model in handle:  # type: Model
            _extract_graph_from_model(model, ordered_nodes, seen_nodes, edges, label_map)
            # Stop after first model
            break

    if not ordered_nodes and not edges:
        print(
            "Warning: No node/1 or edge/2 facts were found in the first answer set.",
            file=sys.stderr,
        )

    return ordered_nodes, edges, label_map


def _extract_graph_from_model(
    model: Model,
    ordered_nodes: List[str],
    seen_nodes: Set[str],
    edges: Set[Tuple[str, str]],
    label_map: Dict[str, str],
) -> None:
    """
    Given a clingo Model, extract node/1, edge/2, and node_label/2 facts
    and add them to the provided containers.

    - Nodes are stored in ordered_nodes in the order they first appear.
    - seen_nodes is used to avoid duplicates.
    - Edges are directed.
    """
    for sym in model.symbols(shown=True):
        name = sym.name
        args = sym.arguments

        if name == "node" and len(args) == 1:
            node_id = str(args[0])
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                ordered_nodes.append(node_id)

        elif name == "edge" and len(args) == 2:
            u = str(args[0])
            v = str(args[1])
            edges.add((u, v))

        elif name == "node_label" and len(args) == 2:
            node_id = str(args[0])
            label = str(args[1])
            label_map[node_id] = label


# ---------------------------------------------------------------------------
# Isolated node filtering
# ---------------------------------------------------------------------------

def filter_isolated_nodes(
    ordered_nodes: List[str],
    edges: Set[Tuple[str, str]],
    label_map: Dict[str, str],
) -> Tuple[List[str], Set[Tuple[str, str]], Dict[str, str]]:
    """
    Remove isolated nodes (nodes that do not occur in any edge).

    Args:
        ordered_nodes: list of all node identifiers in ID order.
        edges:         set of directed edges (u, v).
        label_map:     node_id -> label mapping.

    Returns:
        filtered_nodes: list of node identifiers that appear in at least one edge.
        filtered_edges: unchanged edges (edges already only use non-isolated nodes).
        filtered_labels: labels restricted to filtered_nodes.

    If there are no edges, the result will have an empty node list.
    """
    if not edges:
        print("Warning: No edges found; all nodes are isolated and will be removed.", file=sys.stderr)
        return [], set(), {}

    used_nodes: Set[str] = set()
    for u, v in edges:
        used_nodes.add(u)
        used_nodes.add(v)

    filtered_nodes = [n for n in ordered_nodes if n in used_nodes]
    filtered_labels = {n: label_map[n] for n in filtered_nodes if n in label_map}

    removed = len(ordered_nodes) - len(filtered_nodes)
    if removed > 0:
        print(f"Filtered out {removed} isolated node(s).", file=sys.stderr)

    return filtered_nodes, edges, filtered_labels


# ---------------------------------------------------------------------------
# Adjacency matrix construction and CSV output
# ---------------------------------------------------------------------------

def build_adjacency_matrix(
    ordered_nodes: List[str],
    edges: Set[Tuple[str, str]],
) -> List[List[int]]:
    """
    Build an adjacency matrix from an ordered list of node identifiers
    and a set of directed edges.

    Args:
        ordered_nodes: list of node identifiers (strings), defines ID order
                       ID(node) = index in this list + 1
        edges:         set of directed edges (u, v)

    Returns:
        matrix: a square matrix (list of rows), where matrix[i][j] is 1 if
                there is an edge from node i+1 to node j+1, else 0.
    """
    index_of = {node: i for i, node in enumerate(ordered_nodes)}

    n = len(ordered_nodes)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for (u, v) in edges:
        if u not in index_of or v not in index_of:
            continue
        i = index_of[u]
        j = index_of[v]
        matrix[i][j] = 1

    return matrix


def write_adjacency_csv(
    path: str,
    num_nodes: int,
    matrix: List[List[int]],
) -> None:
    """
    Write the adjacency matrix to a CSV file using integer node IDs.

    CSV format:
        - First row: empty cell followed by node IDs (1..n) as column headers.
        - Subsequent rows: row node ID followed by 0/1 entries.
          (Row i, column j is 1 if there is an edge i -> j.)
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        node_ids = list(range(1, num_nodes + 1))

        # Header row
        header = [""] + node_ids
        writer.writerow(header)

        # Matrix rows
        for node_id, row in zip(node_ids, matrix):
            writer.writerow([node_id] + row)


def write_nodes_mapping_csv(
    path: str,
    ordered_nodes: List[str],
    label_map: Dict[str, str],
) -> None:
    """
    Write node ID → label mapping to a CSV file.

    Label logic:
      - If node_label(Node, Label) was present in the answer set, Label is used.
      - Otherwise, the label is the node's internal identifier (node_id string).

    CSV format:
        id,label
        1,some_label_or_node_id
        2,some_label_or_node_id
        ...
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])

        for idx, node_id in enumerate(ordered_nodes, start=1):
            # Default label is the internal node identifier string
            label = label_map.get(node_id, node_id)
            writer.writerow([idx, label])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main command-line entry point.
    """
    args = parse_args()

    program_source = load_program_source(args)
    meta_program_source = load_meta_program_source(args)

    reified_symbols = reify_program(program_source)
    reified_program = "".join([f"{symbol}.\n" for symbol in reified_symbols])

    ordered_nodes, edges, label_map = combine_and_solve(reified_program, meta_program_source)

    ordered_nodes, edges, label_map = filter_isolated_nodes(ordered_nodes, edges, label_map)


    matrix = build_adjacency_matrix(ordered_nodes, edges)

    write_adjacency_csv(args.output, len(ordered_nodes), matrix)
    write_nodes_mapping_csv(args.nodes_output, ordered_nodes, label_map)

    print(
        f"Wrote adjacency matrix for {len(ordered_nodes)} nodes to {args.output}"
    )
    print(
        f"Wrote node ID-to-label mapping for {len(ordered_nodes)} nodes to {args.nodes_output}"
    )


if __name__ == "__main__":
    main()
