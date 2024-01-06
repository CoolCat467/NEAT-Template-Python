"""NEAT Artificial Intelligence Template Module ported from CodeBullet JavaScript code."""

# Ported from https://github.com/Code-Bullet/NEAT-Template-JavaScript

# Copyright (C) 2023  CoolCat467
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__title__ = "NEAT Artificial Intelligence Template Module"
__author__ = "CoolCat467 & CodeBullet"
__version__ = "3.0.0"
__license__ = "GNU General Public License Version 3"

import json
import math
import random
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self


def sigmoid(value: float) -> float:
    """Return the sigmoid of input.

    AIs use this to prevent the need for keeping track of a threshold value
    for activation, which would make it hard to do learning.
    """
    # return 1 / (1 + (math.exp(-4.9 * value)))#Original
    # No idea why the original does -4.9 * value, that is odd.
    return 1 / (1 + (math.exp(-value)))  # Modified


NodeSave = tuple[int, int, tuple[int, ...]]


class Node:
    """Represents a neuron in a brain."""

    __slots__ = (
        "number",
        "input_sum",
        "output_value",
        "output_connections",
        "layer",
    )

    def __init__(self, number: int) -> None:
        """Initialize Node."""
        self.number: int = number
        # current sum i.e. before activation
        self.input_sum: float = 0.0
        # after activation function is applied
        self.output_value: float = 0.0
        self.output_connections: list[Connection] = []
        self.layer: int = 0

    def __repr__(self) -> str:
        """Return representation of this node."""
        return f"{self.__class__.__name__}({self.number})"

    def engage(self) -> None:
        """Calculate output using sigmoid function, add value to connected node inputs."""
        # The node sends its output to the inputs of the nodes its connected to
        if self.layer != 0:
            # No sigmoid for the inputs and bias
            self.output_value = sigmoid(self.input_sum)

        for conn in self.output_connections:
            if conn.enabled:
                # Add the weighted output to the sum of the inputs of
                # whatever node this node is connected to
                conn.to_node.input_sum += conn.weight * self.output_value

    def is_connected_to(self, node: Node) -> bool:
        """Return True if this node is connected to <node>."""
        if node.layer == self.layer:
            # can't connect if on same layer
            return False

        if node.layer < self.layer:
            for conn in node.output_connections:
                if conn.to_node == self:
                    return True
        else:
            for conn in self.output_connections:
                if conn.to_node == node:
                    return True
        return False

    def __copy__(self) -> Node:
        """Return copy of this node."""
        clone = Node(int(self.number))
        clone.layer = int(self.layer)
        return clone

    clone = __copy__

    def save(self) -> NodeSave:
        """Return a list containing important data about this node."""
        cnodes = []
        for conn in self.output_connections:
            cnodes.append(conn.from_node.number)
            cnodes.append(conn.to_node.number)
        return self.number, self.layer, tuple(cnodes)

    @classmethod
    def load(cls, data: NodeSave) -> Node:
        """Return a Node Object with data initialized from given data input."""
        node = cls(data[0])
        node.layer = data[1]
        # Gets set with connect_nodes in genome class
        node.output_connections = []
        return node


ConnectionSave = tuple[int, int, float, int, bool]


class Connection(NamedTuple):
    """Object representing a connection between two node objects."""

    from_node: Node
    to_node: Node
    weight: float
    # each connection is given a innovation number to compare genomes
    innovation_number: int
    enabled: bool = True

    def mutate_weight(self) -> Self:
        """Mutate the weight of this connection."""
        change = random.randint(1, 10)
        if change == 1:  # 10% of the time completely change the self.weight
            new_weight = random.random() * 2 - 1
        else:
            # otherwise slightly change it
            new_weight = self.weight + random.gauss(0, 1) / 50
        # Keep self.weight within bounds
        return self.__class__(
            from_node=self.from_node,
            to_node=self.to_node,
            weight=max(min(new_weight, 1), -1),
            innovation_number=self.innovation_number,
            enabled=self.enabled,
        )

    def clone(self, from_node: Node, to_node: Node) -> Connection:
        """Return a clone of self, but with potentially different from_node and to_node values."""
        return self.__class__(
            from_node,
            to_node,
            self.weight,
            self.innovation_number,
            self.enabled,
        )

    def save(self) -> ConnectionSave:
        """Return a list containing important information about this connection."""
        return (
            self.from_node.number,
            self.to_node.number,
            self.weight,
            self.innovation_number,
            self.enabled,
        )


HistorySave = tuple[int, int, int, list[int]]


class History(NamedTuple):
    """Object for storing information about the past connections."""

    from_node: int
    to_node: int
    innovation_number: int
    innovation_numbers: list[int]

    # the innovation Numbers from the connections of the
    # genome which first had this mutation
    # our self represents the genome and allows us to test if
    # another genome is the same
    # as our self is before this connection was added

    def __repr__(self) -> str:
        """Return representation of this object."""
        return f"{self.__class__.__name__}({self.from_node}, {self.to_node}, {self.innovation_number}, {self.innovation_numbers})"

    def matches(
        self,
        genome: Genome,
        from_node: Node,
        to_node: Node,
    ) -> bool:
        """Return True if genomes are the same."""
        if len(genome.genes) == len(self.innovation_numbers) and (
            from_node.number == self.from_node
            and to_node.number == self.to_node
        ):
            for gene in genome.genes:
                if gene.innovation_number not in self.innovation_numbers:
                    return False
            # If reached this far then innovation_numbers matches the gene
            # innovation numbers and the connection between the same nodes,
            # so it does match
            return True
        return False

    def clone(self) -> History:
        """Return a clone of self."""
        return History(
            self.from_node,
            self.to_node,
            self.innovation_number,
            deepcopy(self.innovation_numbers),
        )

    def save(self) -> HistorySave:
        """Return a list of important information about this history object."""
        return (
            self.from_node,
            self.to_node,
            self.innovation_number,
            self.innovation_numbers,
        )


def matching_gene(parent: Genome, innovation_number: int) -> int | None:
    """Return index to a gene matching the input innovation number in the input genome."""
    for idx, gene in enumerate(parent.genes):
        if gene.innovation_number == innovation_number:
            return idx
    # no matching gene found
    return None


GenomeSave = tuple[
    int,
    int,
    list[ConnectionSave],
    list[NodeSave],
    int,
    int,
    int,
]


class Genome:
    """Pretty much a brain, but it's called genome."""

    mutate_random = False
    __slots__ = (
        "genes",
        "nodes",
        "inputs",
        "outputs",
        "layers",
        "next_node",
        "bias_node",
        "network",
    )

    def __init__(
        self,
        inputs: int,
        outputs: int,
        crossover: bool = False,
    ) -> None:
        """Initialize Genome."""
        # A list of connections between our nodes which represent the neural network
        self.genes: list[Connection] = []
        self.nodes: list[Node] = []
        self.inputs = inputs
        self.outputs = outputs
        self.layers = 2
        self.next_node = 0
        # A list of nodes in the order that they are needed to be considered in the neural network
        self.network: list[Node] = []
        # create input nodes

        if crossover:
            return

        for i in range(inputs):
            node = Node(i)
            node.layer = 0
            self.nodes.append(node)
            self.next_node += 1

        # create output nodes
        for i in range(outputs):
            self.nodes.append(Node(i + self.inputs))
            self.nodes[i + self.inputs].layer = 1
            self.next_node += 1

        # add bias node
        self.nodes.append(Node(self.next_node))
        self.bias_node = self.next_node
        self.next_node += 1
        self.nodes[self.bias_node].layer = 0

    def __repr__(self) -> str:
        """Return simple representation."""
        return f"<Genome Object: layers: {self.layers} Bias: {self.bias_node} nodes: {len(self.nodes)} genes: {len(self.genes)}>"

    def get_innov_no(
        self,
        innovation_history: list[History],
        from_node: Node,
        to_node: Node,
    ) -> int:
        """Return the innovation number for the new mutation."""
        is_new: bool = True
        conn_innov_no: int = 0
        for innov in innovation_history:
            if innov.matches(self, from_node, to_node):
                is_new = False
                conn_innov_no = innov.innovation_number
                break

        if is_new:
            # if the mutation is new then record current state of the genome
            inno_nos = [gene.innovation_number for gene in self.genes]
            innovation_history.append(
                History(
                    from_node.number,
                    to_node.number,
                    conn_innov_no,
                    inno_nos,
                ),
            )
            conn_innov_no += 1
        return conn_innov_no

    def connect_nodes(self) -> None:
        """Ensure all nodes know about each other so feed_forward works correctly."""
        # Clear connections
        for node in self.nodes:
            node.output_connections.clear()
        # For each connection, add the corresponding gene to the node.
        for gene in self.genes:
            gene.from_node.output_connections.append(gene)

    def fully_connect(self, innovation_history: list[History]) -> None:
        """Connect all nodes to each other."""
        for inode in (self.nodes[i] for i in range(self.inputs)):
            for onode in (
                self.nodes[len(self.nodes) - ii - 2]
                for ii in range(self.outputs)
            ):
                conn_innov_no = self.get_innov_no(
                    innovation_history,
                    inode,
                    onode,
                )
                self.genes.append(
                    Connection(
                        inode,
                        onode,
                        random.random() * 2 - 1,
                        conn_innov_no,
                    ),
                )
        bias = self.nodes[self.bias_node]
        conn_innov_no = self.get_innov_no(
            innovation_history,
            bias,
            self.nodes[len(self.nodes) - 2],
        )
        self.genes.append(
            Connection(
                bias,
                self.nodes[len(self.nodes) - 2],
                random.random() * 2 - 1,
                conn_innov_no,
            ),
        )

        conn_innov_no = self.get_innov_no(
            innovation_history,
            bias,
            self.nodes[len(self.nodes) - 3],
        )
        self.genes.append(
            Connection(
                bias,
                self.nodes[len(self.nodes) - 3],
                random.random() * 2 - 1,
                conn_innov_no,
            ),
        )

        self.connect_nodes()

    def get_node(self, node_no: int) -> Node:
        """Return the node with a matching number, as sometimes self.nodes will not be in order."""
        for node in self.nodes:
            if node.number == node_no:
                return node
        raise LookupError(f"Node with ID {node_no!r} does not exist!")

    def feed_forward(self, input_values: Iterable[float]) -> tuple[float, ...]:
        """Feed in input values for the neural network and return output."""
        # Set the outputs of the input nodes
        for idx, value in enumerate(input_values):
            self.nodes[idx].output_value = value

        self.nodes[self.bias_node].output_value = 1  # Output of bias is 1

        for node in self.network:
            node.engage()

        # the outputs are the self.nodes[inputs] to self.nodes[inputs+outputs-1]
        outs = {
            i: self.nodes[self.inputs + 1].output_value
            for i in range(self.outputs)
        }

        # reset all nodes for the next feed forward
        for node in self.nodes:
            node.input_sum = 0

        return tuple(outs[k] for k in sorted(outs.keys()))

    def generate_network(self) -> list[Node]:
        """Set up the neural network as a list of nodes in order to be engaged."""
        self.connect_nodes()
        self.network = []
        # For each layer add the node in that layer,
        # since layers cannot connect to themselves there is no need
        # to order the nodes within a layer
        for layer in range(self.layers):
            for node in self.nodes:
                # If the node is in that layer
                if node.layer == layer:
                    # Add that node to the network
                    self.network.append(node)
        return self.network

    def fully_connected(self) -> bool:
        """Return whether the network is fully connected or not."""
        max_conns = 0
        # Dictionary which stored the amount of nodes in each layer
        nodes_in_layers = {layer: 0 for layer in range(self.layers)}
        # Populate dictionary
        for node in self.nodes:
            nodes_in_layers[node.layer] += 1

        # for each layer the maximum amount of connections is the number of the
        # layer times the number of nodes in front of it.
        # so lets add the max for each layer together and then we will get
        # the maximum amount of connections in the network
        for layer in range(self.layers):
            nodes_in_front = 0
            # for i in range(layer + 1, self.layers):# for each layer in front of this layer,
            for i in range(layer + 1, self.layers):
                nodes_in_front += nodes_in_layers[i]  # add up nodes

            max_conns += nodes_in_layers[layer] * nodes_in_front
        # if the number of connections is equal to the max number of
        # connections possible then it is full
        return max_conns <= len(self.genes)

    def random_conn_nodes_bad(self, rn_1: int, rn_2: int) -> bool:
        """Return True if the two given nodes connected to each other."""
        if self.nodes[rn_1].layer == self.nodes[rn_2].layer:
            # if the nodes are in the same layer
            return True
        if self.nodes[rn_1].is_connected_to(self.nodes[rn_2]):
            # if the nodes are already connected
            return True
        return False

    def add_conn(self, innovation_history: list[History]) -> None:
        """Add a new connection between two nodes that aren't currently connected."""
        # Cannot add a connection to a fully connected network
        if self.fully_connected():
            print("Connection failed.")
            raise RuntimeError(
                "Cannot add a connection to a fully connected network.",
            )

        # Get random node
        random_node1 = random.randrange(len(self.nodes))
        random_node2 = random.randrange(len(self.nodes))
        # If the nodes are the same or are connected, get new random nodes
        while self.random_conn_nodes_bad(random_node1, random_node2):
            random_node1 = random.randrange(len(self.nodes))
            random_node2 = random.randrange(len(self.nodes))
        # if the first random node is after the second, then switch them
        if self.nodes[random_node1].layer > self.nodes[random_node2].layer:
            random_node1, random_node2 = random_node2, random_node1
        # get the innovation number of the connection
        # this will be a new number if no identical genome has mutated in the same way
        conn_innov_no = self.get_innov_no(
            innovation_history,
            self.nodes[random_node1],
            self.nodes[random_node2],
        )

        # Add the connection with a random dictionary
        self.genes.append(
            Connection(
                self.nodes[random_node1],
                self.nodes[random_node2],
                random.random() * 2 - 1,
                conn_innov_no,
            ),
        )
        self.connect_nodes()

    def add_node(self, innovation_history: list[History]) -> None:
        """Pick a random connection to create a node between."""
        if not self.genes:
            self.add_conn(innovation_history)

        random_conn = random.choice(self.genes)
        if len(self.genes) != 1:
            while random_conn.from_node == self.nodes[self.bias_node]:
                random_conn = random.choice(self.genes)

        random_conn.enabled = False  # Disable it

        new_node_no = int(self.next_node)
        self.nodes.append(Node(new_node_no))
        self.next_node += 1

        # Add a new connection to the new node with a weight of 1
        new_node = self.get_node(new_node_no)
        conn_innov_no = self.get_innov_no(
            innovation_history,
            random_conn.from_node,
            new_node,
        )
        self.genes.append(
            Connection(random_conn.from_node, new_node, 1, conn_innov_no),
        )

        conn_innov_no = self.get_innov_no(
            innovation_history,
            new_node,
            random_conn.to_node,
        )
        new_node.layer = random_conn.from_node.layer + 1

        conn_innov_no = self.get_innov_no(
            innovation_history,
            self.nodes[self.bias_node],
            new_node,
        )
        # connect the bias to the new node with a weight of 0
        self.genes.append(
            Connection(self.nodes[self.bias_node], new_node, 0, conn_innov_no),
        )

        # If the layer of the new node is equal to the layer of the output node
        # of the old connection, then the new layer needs the be created more
        # accurately the layer numbers of all layers equal to or greater than
        # this new node need to be incremented
        if new_node.layer == random_conn.to_node.layer:
            for node in self.nodes[:-1]:
                if node.layer >= new_node.layer:
                    node.layer += 1
            self.layers += 1
        self.connect_nodes()

    def mutate(self, innovation_history: list[History]) -> None:
        """Mutates the genome in random ways."""
        if not self.genes:
            self.add_conn(innovation_history)
        if self.mutate_random:
            if random.randint(0, 9) < 8:  # 80% of the time mutate weights
                for index, gene in enumerate(tuple(self.genes)):
                    self.genes[index] = gene.mutate_weight()
            if (
                random.randint(0, 99) < 5
            ):  # 5% of the time add a new connection
                self.add_conn(innovation_history)

            if random.randint(0, 99) == 0:  # 1% of the time add a node
                self.add_node(innovation_history)
            return
        # Do all mutations
        for index, gene in enumerate(tuple(self.genes)):
            self.genes[index] = gene.mutate_weight()
        self.add_conn(innovation_history)
        self.add_node(innovation_history)

    def crossover(self, parent: Genome) -> Genome:
        """Return new genome by combining this genome with another."""
        child = self.__class__(int(self.inputs), int(self.outputs), True)
        child.genes = []
        child.nodes = []
        child.layers = int(self.layers)
        child.next_node = int(self.next_node)
        child.bias_node = int(self.bias_node)
        # list of genes to be inherited from the parents
        child_genes = []
        is_enabled = []
        # All inherited genes
        for gene in self.genes:
            set_enabled = True
            parent_gene = matching_gene(parent, gene.innovation_number)
            if parent_gene is not None:
                # if the genes match
                if (
                    not gene.enabled
                    or not parent.genes[parent_gene].enabled
                    and random.randint(1, 4) < 3
                ):
                    # 75% of the time disable the child's gene
                    set_enabled = False
                rand = random.randint(0, 100)
                if rand > 5:
                    # Get gene from ourselves, we are better
                    # by the way original was <. odd.
                    child_genes.append(gene)
                else:  # Otherwise, get gene from parent 2 which is worse
                    child_genes.append(parent.genes[parent_gene])
            else:  # disjoint or excess gene
                child_genes.append(gene)
                set_enabled = gene.enabled
            is_enabled.append(set_enabled)

        # since all excess and disjoint genes are inherited from the more
        # fit parent (this Genome) the child's structure is no different from
        # this parent, with exception of dormant connections being enabled but
        # this won't effect our nodes
        # so all the nodes can be inherited from this parent
        for node in self.nodes:
            child.nodes.append(node.clone())

        # Clone all the connections so that they connect the child's new nodes
        for idx, connection in enumerate(child_genes):
            child.genes.append(
                connection.clone(
                    child.get_node(gene.from_node.number),
                    child.get_node(gene.to_node.number),
                ),
            )
            connection.enabled = is_enabled[idx]

        child.connect_nodes()
        return child

    def print_geneome(self) -> None:
        """Print out information about genome."""
        print("Private genome layers:", self.layers)
        print("Bias node:", self.bias_node)
        print("Nodes:")
        print(", ".join(str(node.number) for node in self.nodes))
        print("Genes:")
        for gene in self.genes:
            # for each Connection
            print(
                f"Gene {gene.innovation_number} From node {gene.from_node.number} "
                f"To node {gene.to_node.number} is enabled:{gene.enabled} "
                "from layer {gene.from_node.layer} to layer {gene.to_node.layer} "
                "weight: {gene.weight}",
            )
        print()

    def __copy__(self) -> Genome:
        """Return a copy of this genome."""
        clone = Genome(self.inputs, self.outputs, True)
        # Copy our nodes
        for node in self.nodes:
            clone.nodes.append(node.clone())
        # Copy all the connections so that they connect to the clone's new nodes
        for gene in self.genes:
            from_node = clone.get_node(gene.from_node.number)
            to_node = clone.get_node(gene.to_node.number)
            assert from_node is not None
            assert to_node is not None
            clone.genes.append(gene.clone(from_node, to_node))

        clone.layers = int(self.layers)
        clone.next_node = int(self.next_node)
        clone.bias_node = int(self.bias_node)
        clone.connect_nodes()

        return clone

    clone = __copy__

    def save(self) -> GenomeSave:
        """Return important information about this Genome Object."""
        genes = [gene.save() for gene in self.genes]
        nodes = [node.save() for node in self.nodes]
        return (
            self.inputs,
            self.outputs,
            genes,
            nodes,
            self.layers,
            self.next_node,
            self.bias_node,
        )

    @classmethod
    def load(cls, data: GenomeSave) -> Genome:
        """Return a new Genome Object based on save data input."""
        self = cls(*data[:2], False)
        (
            self.inputs,
            self.outputs,
            genes,
            nodes,
            self.layers,
            self.next_node,
            self.bias_node,
        ) = data
        self.nodes = [Node.load(i) for i in nodes]
        tmpgenes = [
            (
                Connection(
                    self.get_node(frm),
                    self.get_node(to),
                    weight,
                    inno,
                ),
                enabled,
            )
            for frm, to, weight, inno, enabled in genes
        ]
        self.genes = []
        for connection, enabled in tmpgenes:
            connection.enabled = bool(enabled)
            self.genes.append(connection)
        # self.connect_nodes() already called in generate_network
        self.generate_network()
        return self


PlayerSave = tuple[int, GenomeSave]


Move = TypeVar("Move")


class BasePlayer(Generic[Move], metaclass=ABCMeta):
    """Base class for a player object. Many functions simply pass instead of doing stuff."""

    __slots__ = ("gen", "brain")

    def __init__(
        self,
        generation: int = 0,
        brain: Genome | None = None,
    ) -> None:
        """Initialize BasePlayer."""
        self.gen = generation

        if brain is None:
            self.brain = self.start()
        else:
            self.brain = brain

    @abstractmethod
    def start(self) -> Genome:
        """Return new brain."""

    @abstractmethod
    def look(self, *args: object) -> Iterable[float]:
        """Return inputs for the nural network."""

    def think(self, inputs: Iterable[float]) -> Iterable[float]:
        """Return decision from nural network."""
        return self.brain.feed_forward(inputs)

    @abstractmethod
    def interpret(self, decision: Iterable[float]) -> Move:
        """Interpret the outputs from the neural network."""

    @abstractmethod
    def update(self) -> None:
        """Do whatever actions are required for one tick of game."""

    @abstractmethod
    def calculate_fitness(self) -> float:
        """Calculate the fitness of the AI."""

    @abstractmethod
    def is_dead(self) -> bool:
        """Return if player is dead."""

    def simulate(self, *look_args: object) -> Move:
        """Look, think, and update."""
        # Get inputs for nural network
        inputs = self.look(*look_args)
        # Get nural network's decision (output) from inputs
        decision = self.think(inputs)
        # Move the player according to the outputs from the
        # neural network
        return self.interpret(decision)

    def clone(self) -> Self:
        """Return a clone of self."""
        brain_copy = self.brain.clone()
        brain_copy.generate_network()
        return self.__class__(generation=self.gen, brain=brain_copy)

    def crossover(self, parent: BasePlayer[object]) -> Self:
        """Return a BasePlayer object by crossing over our brain and parent2's brain."""
        child_brain = self.brain.crossover(parent.brain)
        child_brain.generate_network()
        return self.__class__(self.gen + 1, child_brain)

    def save(self) -> PlayerSave:
        """Return a list containing important information about ourselves."""
        return (
            self.gen,
            self.brain.save(),
        )

    @classmethod
    def load(cls, data: PlayerSave) -> Self:
        """Return a BasePlayer Object with save data given."""
        gen, brain_data = data
        brain = Genome.load(brain_data)
        return cls(gen, brain)


SpeciesSave = tuple[
    list[PlayerSave],
    float,
    PlayerSave,
    int,
    float,
    float,
    float,
]


Player = TypeVar("Player", bound=BasePlayer[object], covariant=True)


class Species(Generic[Player]):
    """Species object, containing large groups of players."""

    __slots__ = (
        "players",
        "best_fitness",
        "champ",
        "staleness",
        "rep",
        "excess_coeff",
        "w_diff_coeff",
        "compat_threshold",
    )

    def __init__(self, player: Player | None = None) -> None:
        """Initialize Species object."""
        self.players = []
        self.best_fitness = 0.0
        self.champ: Player
        self.staleness = 0
        # how many generations have gone without an improvement
        self.rep: Genome

        # Coefficients for testing compatibility
        self.excess_coeff = 1.0
        self.w_diff_coeff = 0.5
        self.compat_threshold = 3.0
        if player:
            self.players.append(player)
            # Since it is the only one in the species it is by default the best
            self.best_fitness = player.calculate_fitness()
            self.rep = player.brain.clone()
            self.champ = player.clone()

    def __repr__(self) -> str:
        """Return what this object should be represented by in the python interpreter."""
        return "<Species Object>"

    @staticmethod
    def get_excess_disjoint(brain1: Genome, brain2: Genome) -> int | float:
        """Return the number of excess and disjoint genes."""
        matching = 0
        for gene1 in brain1.genes:
            for gene2 in brain2.genes:
                if gene1.innovation_number == gene2.innovation_number:
                    matching += 1
                    break
        # Return number of excess and disjoint genes
        return (len(brain1.genes) + len(brain2.genes) - 2) * matching

    @staticmethod
    def avg_w_diff(brain1: Genome, brain2: Genome) -> int | float:
        """Return the average weight difference between two brains."""
        if not brain1.genes or not brain2.genes:
            return 0
        matching = 0
        total_diff = 0.0
        for gene1 in brain1.genes:
            for gene2 in brain2.genes:
                if gene1.innovation_number == gene2.innovation_number:
                    matching += 1
                    total_diff += abs(gene1.weight - gene2.weight)
                    break
        ##        if not matching:
        ##            return 100#divide by zero error otherwise
        return 100 if not matching else total_diff / matching

    def same_species(self, genome: Genome) -> bool:
        """Return if a genome is in this species."""
        excess_and_disjoint = self.get_excess_disjoint(genome, self.rep)
        avg_w_diff = self.avg_w_diff(genome, self.rep)
        large_genome_normalizer = max(len(genome.genes) - 20, 1)
        # compatibility formula
        compatibility = (
            self.excess_coeff * excess_and_disjoint / large_genome_normalizer
        ) + (self.w_diff_coeff * avg_w_diff)
        return self.compat_threshold > compatibility

    def add_to_species(self, player: Player) -> None:
        """Add given player to this species."""
        self.players.append(player)

    def calculate_fitnesses(self) -> dict[int, float]:
        """Calculate the fitness of each player."""
        fitnesses = {}
        for index, player in enumerate(self.players):
            fitnesses[index] = player.calculate_fitness()
        return fitnesses

    def sort_species(self) -> None:
        """Sort the species by their fitness."""
        fitnesses = self.calculate_fitnesses()
        order = sorted(
            fitnesses.items(),
            key=lambda data: data[1],
            reverse=True,
        )
        new_players = []
        for index, _fitness in order:
            new_players.append(self.players[index])

        self.players = new_players
        if not self.players:
            self.staleness = 200
            return
        # if new best player
        if order[0][1] > self.best_fitness:
            self.staleness = 0
            self.best_fitness = order[0][1]
            # self.rep = self.players[0].clone()
            self.rep = self.players[order[0][0]].brain.clone()
        else:  # If no new best player,
            self.staleness += 1
        return

    @property
    def average_fitness(self) -> float:
        """Calculates the average fitness of this species."""
        if not self.players:
            return 0.0
        return sum(p.calculate_fitness() for p in self.players) / len(
            self.players,
        )

    def select_player(self) -> Player:
        """Select a player based on it's fitness."""
        if len(self.players) == 0:
            raise RuntimeError("No players!")
        fitness_sum = math.floor(
            sum(player.calculate_fitness() for player in self.players),
        )
        rand = 0
        if fitness_sum > 0:
            rand = random.randrange(fitness_sum)
        running_sum = 0.0
        for player in self.players:
            running_sum += player.calculate_fitness()
            if running_sum > rand:
                return player
        return self.players[0]

    def give_me_baby(self, innovation_history: list[History]) -> Player:
        """Return a baby from either random clone or crossover of bests."""
        if random.randint(0, 3) == 0:
            # 25% of the time there is no crossover and child is a clone of a semi-random player
            baby = self.select_player().clone()
        else:
            parent1 = self.select_player()
            parent2 = self.select_player()

            # The crossover function expects the highest fitness parent
            # to be the object and the second parent as the argument
            if parent1.calculate_fitness() < parent2.calculate_fitness():
                parent1, parent2 = [parent2, parent1]
            baby = parent1.crossover(parent2)
        baby.brain.mutate(innovation_history)
        return baby

    def cull(self) -> None:
        """Kill half of the players."""
        if len(self.players) > 2:
            self.players = self.players[int(len(self.players) // 2) :]

    def clone(self) -> Self:
        """Return a clone of self."""
        clone = self.__class__()
        clone.players = [player.clone() for player in self.players]
        clone.best_fitness = float(self.best_fitness)
        clone.champ = self.champ.clone()
        clone.staleness = int(self.staleness)
        clone.excess_coeff = float(self.excess_coeff)
        clone.w_diff_coeff = float(self.w_diff_coeff)
        clone.compat_threshold = float(self.compat_threshold)
        return clone

    def __copy__(self) -> Self:
        """Return a copy of self."""
        return self.clone()

    def save(self) -> SpeciesSave:
        """Return a list containing important information about this species."""
        players = [player.save() for player in self.players]
        champ = self.champ.save()
        ##        rep = self.rep.save()
        return (
            players,
            self.best_fitness,
            champ,
            self.staleness,
            self.excess_coeff,
            self.w_diff_coeff,
            self.compat_threshold,
        )


PopulationSave = tuple[
    list[PlayerSave],
    float,
    int,
    list[HistorySave],
    list[SpeciesSave],
]


class Population(Generic[Player]):
    """Population Object, stores groups of species."""

    __slots__ = (
        "player_class",
        "gen",
        "players",
        "innovation_history",
        "species",
        "best_fitness",
    )

    def __init__(
        self,
        player_class: type[Player],
        size: int,
        add_players: bool = True,
    ) -> None:
        """Initialize population object."""
        self.player_class = player_class

        self.gen = 0

        self.players = []
        self.innovation_history: list[History] = []
        self.species: list[Species[Player]] = []
        self.best_fitness = 0.0

        if add_players:
            for _ in range(size):
                self.players.append(self.player_class())
                self.players[-1].brain.mutate(self.innovation_history)
                self.players[-1].brain.generate_network()

    def __repr__(self) -> str:
        """Representation of Population."""
        return f"<Population Object with {len(self.players)} Players and {self.gen} Generations>"

    def update_alive(self) -> None:
        """Update all of the players that are alive."""
        for player in list(self.players):
            if not player.is_dead():
                player.update()

    def all_dead(self) -> bool:
        """Return True if all the players are dead. :(."""
        return all(player.is_dead() for player in self.players)

    def set_best_player(self) -> None:
        """Set the best player globally and for current generation."""
        if not (self.species and self.species[0].players):
            return
        temp_best = self.species[0].players[0]
        temp_best.gen = self.gen

        current_fitness = temp_best.calculate_fitness()

        if current_fitness >= self.best_fitness:
            self.best_fitness = current_fitness

    def seperate_species(self) -> None:
        """Separate players into species.

        Split based on how similar they are to the leaders of the species
        in the previous generation.
        """
        # Empty current species
        for specie in self.species:
            del specie.players[:]
        # For each player,
        for player in self.players:
            species_found = False
            # For each species
            for specie in self.species:
                if specie.same_species(player.brain):
                    specie.add_to_species(player)
                    species_found = True
                    break
            if not species_found:
                self.species.append(Species(player))

    def calculate_fitnesses(self) -> dict[int, float]:
        """Calculate the fitness of each player."""
        fitnesses = {}
        for index, player in enumerate(self.players):
            fitnesses[index] = player.calculate_fitness()
        return fitnesses

    def sort_species(self) -> None:
        """Sort the species to be ranked in fitness order, best first."""
        for species in self.species:
            species.sort_species()
        # Sort the species by a fitness of its best player
        # using selection sort like a loser
        temp = []
        for _ in range(len(self.species)):
            smax = 0.0
            max_idx = 0
            for idx, specie in enumerate(self.species):
                if specie.best_fitness > smax:
                    smax = specie.best_fitness
                    max_idx = idx
            temp.append(self.species[max_idx])
            del self.species[max_idx]
        self.species = temp

    def mass_extinction(self) -> None:
        """For all the species but the top five, kill them all."""
        for species in range(5, len(self.species)):
            del self.species[species]

    def cull_species(self) -> None:
        """Kill off the bottom half of each species."""
        for species in self.species:
            species.cull()

    def kill_stale_species(self) -> None:
        """Kills all species which haven't improved in 15 generations."""
        for i in range(len(self.species) - 1, -1, -1):
            if self.species[i].staleness >= 15:
                del self.species[i]

    def get_avg_fitness_sum(self) -> int | float:
        """Return the sum of the average fitness for each species."""
        return sum(s.average_fitness for s in self.species)

    def kill_bad_species(self) -> None:
        """Kill species which are so bad they can't reproduce."""
        average_sum = self.get_avg_fitness_sum()
        if not average_sum:
            return
        for i in range(len(self.species) - 1, -1, -1):
            if (
                self.species[i].average_fitness
                / average_sum
                * len(self.players)
                < 1
            ):
                del self.species[i]
        return

    def natural_selection(self) -> None:
        """Generate new generation."""
        previous_best = self.players[0]
        # Separate players into species
        self.seperate_species()
        # Sort the species to be ranked in fitness order, best first
        self.sort_species()
        # Kill off the bottom half of each species
        self.cull_species()
        # # Save the best player of this generation
        self.set_best_player()
        # Remove species which haven't improved in 15 generations
        self.kill_stale_species()
        # Kill species which are super bad
        self.kill_bad_species()

        average_sum = self.get_avg_fitness_sum()
        if average_sum == 0:
            average_sum = 0.1
        children = []
        for species in self.species:
            # Add champion without any mutation
            children.append(species.champ.clone())
            child_count = round(
                (species.average_fitness / average_sum * len(self.players))
                - 1,
            )
            for _ in range(child_count):
                children.append(species.give_me_baby(self.innovation_history))
        if len(children) < len(self.players):
            children.append(previous_best.clone())
        # If not enough babies
        while len(children) < len(self.players):
            if self.species:
                # Get babies from the past generation
                children.append(
                    self.species[0].give_me_baby(self.innovation_history),
                )
            else:
                clone = previous_best.clone()
                clone.brain.mutate(self.innovation_history)
                children.append(clone)

        self.players = children
        self.gen += 1
        for player in self.players:
            player.brain.generate_network()

    def clone(self) -> Self:
        """Return a clone of self."""
        clone = self.__class__(self.player_class, len(self.players))
        clone.players = [player.clone() for player in self.players]
        clone.best_fitness = float(self.best_fitness)
        clone.gen = int(self.gen)
        clone.innovation_history = [
            ih.clone() for ih in self.innovation_history
        ]
        clone.species = [sep.clone() for sep in self.species]
        return clone

    def __copy__(self) -> Self:
        """Return a copy of self."""
        return self.clone()

    def save(self) -> PopulationSave:
        """Return a list containing all important data."""
        players = [player.save() for player in self.players]
        innoh = [innohist.save() for innohist in self.innovation_history]
        species = [specie.save() for specie in self.species]
        return (
            players,
            self.best_fitness,
            self.gen,
            innoh,
            species,
        )

    @classmethod
    def load(
        cls,
        data: PopulationSave,
        player_class: type[Player],
    ) -> Population[Player]:
        """Return Population Object using save data."""
        self: Population[Player] = cls(player_class, len(data[0]), False)
        (
            players,
            self.best_fitness,
            self.gen,
            innoh,
            species,
        ) = data
        self.players = [self.player_class.load(pdat) for pdat in players]
        self.innovation_history = [History(*i) for i in innoh]
        return self


def save(data: Any, filename: str) -> None:
    """Save data to a file."""
    with open(filename, "w", encoding="utf-8") as save_file:
        json.dump(data, save_file)
        save_file.close()


def load(filename: str) -> Any:
    """Return data retrieved from a file."""
    data = []
    with open(filename, encoding="utf-8") as load_file:
        data = json.load(load_file)
        load_file.close()
    return data


def run() -> None:
    """Run example."""
    print("Starting example.")
    filename = "AI_Data.json"
    try:
        data: PopulationSave = load(filename)
    except FileNotFoundError:
        pop = Population(BasePlayer, 5)
    else:
        print(f"Loading population from file {filename!r}")
        pop = Population.load(data, BasePlayer)
        print(f"AI Data loaded from {filename!r}")
    print(pop)
    print("Running Natural Selection program 100 times...")
    for i in range(100):
        pop.natural_selection()
        pop.update_alive()
        print(i)
    print("Natural Selection done.")
    print("Saving AI data to AI_Data.json...")
    save(pop.save(), "AI_Data.json")
    print("Done.")


if __name__ == "__main__":
    run()
