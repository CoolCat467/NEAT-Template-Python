"""Checkers Nural Network Minimax AI."""

# Programmed by CoolCat467

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

__title__ = "Checkers Nural Network Minimax AI"
__author__ = "CoolCat467"
__version__ = "0.0.0"
__license__ = "GNU General Public License Version 3"


from typing import TYPE_CHECKING, Final

import checkers
import minimax
from neat import neat

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

BOARD_SIZE: Final = (8, 8)
MINIMAX: Final = checkers.CheckersMinimax()
DEPTH: Final = 3
PIECE_LOOKUP: Final = {
    0: -0.5,
    1: 0.5,
    2: -1,
    3: 1,
}
MAX_TURNS: Final = 160


def state_to_input(state: checkers.State) -> Generator[float, None, None]:
    """Yield nural network inputs from checkers game state."""
    width, height = BOARD_SIZE
    for y in range(height):
        for i in range(width // 2):
            x = (i << 1) + (y % 2)
            value = state.pieces.get((x, y))
            if value is None:
                yield 0
                continue
            yield PIECE_LOOKUP[value]


##class AIMinimax(checkers.CheckersMinimax):
##    """Nural Network Minimax AI."""
##
##    __slots__ = ("brain",)
##
##    def __init__(self, brain: neat.Genome | None = None) -> None:
##        """Initialize Nural Network Minimax AI."""
##        if brain is None:
##            self.brain = self.start()
##        else:
##            self.brain = brain
##
##    def start(self) -> neat.Genome:
##        """Return new brain."""
##        width, height = BOARD_SIZE
##        positions = (width // 2) * height
##        return neat.Genome(positions, 1)
##
##    def simulate(self, state: checkers.State) -> float:
##        """Return simulated nural network output."""
##        input_ = tuple(state_to_input(state))
##        result = self.brain.feed_forward(input_)
##        return result[0]
##
##    def value(self, state: checkers.State) -> int | float:
##        """Return value of given game state."""
##        original = checkers.CheckersMinimax.value(state)
##        if isinstance(original, int):
##            return original
##        return self.simulate(state)


class Player(neat.BasePlayer[int | float], checkers.CheckersMinimax):
    """Nural network checkers minimax AI player."""

    __slots__ = ("state", "turns")

    def __init__(
        self,
        generation: int = 0,
        brain: neat.Genome | None = None,
    ) -> None:
        """Initialize Player."""
        neat.BasePlayer.__init__(self, generation, brain)
        checkers.CheckersMinimax.__init__(self)

        self.state = checkers.State(
            BOARD_SIZE,
            1,
            checkers.generate_pieces(*BOARD_SIZE),
        )
        self.turns = 0

    def start(self) -> neat.Genome:
        """Initialize new brain."""
        width, height = BOARD_SIZE
        positions = (width // 2) * height
        brain = neat.Genome(positions, 1)
        brain.fully_connect([])
        return brain

    def look(self, *args: object) -> Iterable[float]:
        """Return nural network inputs from game state."""
        state = args
        assert isinstance(state, checkers.State)
        return state_to_input(state)

    def interpret(self, decision: Iterable[float]) -> int | float:
        """Interpret nural network output."""
        return next(iter(decision))

    def value(self, state: checkers.State) -> int | float:
        """Return value of given game state."""
        original = MINIMAX.value(state)
        if isinstance(original, int):
            return original * 2
        return self.simulate(state)

    def update(self) -> None:
        """Either make a move or have computer move."""
        # print(self.state)
        if MINIMAX.player(self.state) == minimax.Player.MIN:
            # print("Minimax Turn")
            value, action = MINIMAX.minimax(self.state, DEPTH)
        else:
            # print("Nural Network Turn")
            value, action = self.minimax(self.state, DEPTH)
        # print(f'{value = }')
        # print(f'{action = }')
        assert action is not None
        self.state = MINIMAX.result(self.state, action)
        self.turns += 1

    def is_dead(self) -> bool:
        """Is dead?."""
        return self.terminal(self.state) or (self.turns > MAX_TURNS)

    def calculate_fitness(self) -> int:
        """Fitness function for natural selection."""
        print(f"{self.turns = }")
        return (0 + self.turns) + round(1000 * MINIMAX.value(self.state))


def run() -> None:
    """Run test of module."""
    ##    player = Player()
    ##    while not player.dead:
    ##        player.update()
    ##    print(f'{player.calculate_fitness() = }')'
    filename = "AI_Data.json"

    try:
        data: neat.PopulationSave = tuple(neat.load(filename))
    except FileNotFoundError:
        population = neat.Population(Player, 20)  # type: ignore[type-var]
    else:
        print(f"Loading population from file {filename!r}")
        population = neat.Population.load(data, Player)  # type: ignore[type-var]
        print(f"AI Data loaded from {filename!r}")

    print(f"{len(population.players) = }")
    while True:
        print(f"\nGeneration {population.gen}")
        while not population.all_dead():
            population.update_alive()
            print(
                f"State at turn {population.players[0].turns}:\n{population.players[0].state}\n",
            )
            # print()
            # print("tick")
        population.natural_selection()
        print(f"{population.best_fitness = }")
        print("Saving Data...")
        data = population.save()
        neat.save(data, "AI_Data.json")
        print("Save complete.")


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.\n")
    run()
