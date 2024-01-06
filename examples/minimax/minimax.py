"""Minimax - Boilerplate code for Minimax AIs."""

# Programmed by CoolCat467

from __future__ import annotations

# Minimax - Boilerplate code for Minimax AIs.
# Copyright (C) 2023  CoolCat467
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__title__ = "Minimax"
__author__ = "CoolCat467"
__version__ = "0.0.0"
__license__ = "GNU General Public License Version 3"

from abc import ABC, abstractmethod
from enum import Enum, auto
from math import inf as infinity
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable


class Player(Enum):
    """Enum for player status."""

    __slots__ = ()

    MIN = auto()
    MAX = auto()


State = TypeVar("State")
Action = TypeVar("Action")


class MinimaxResult(NamedTuple, Generic[Action]):
    """Minimax Result."""

    value: int | float
    action: Action | None


class Minimax(ABC, Generic[State, Action]):
    """Base class for Minimax AIs."""

    __slots__ = ()

    @abstractmethod
    def value(self, state: State) -> int | float:
        """Return the value of a given game state."""

    @classmethod
    @abstractmethod
    def terminal(cls, state: State) -> bool:
        """Return if given game state is terminal."""

    @classmethod
    @abstractmethod
    def player(cls, state: State) -> Player:
        """Return player status given the state of the game.

        Must return either Player.MIN or Player.MAX
        """

    @classmethod
    @abstractmethod
    def actions(cls, state: State) -> Iterable[Action]:
        """Return a collection of all possible actions in a given game state."""

    @classmethod
    @abstractmethod
    def result(cls, state: State, action: Action) -> State:
        """Return new game state after performing action on given state."""

    def minimax(
        self,
        state: State,
        depth: int | None = 5,
    ) -> MinimaxResult[Action]:
        """Return minimax result best action for a given state for the current player."""
        if self.terminal(state):
            return MinimaxResult(self.value(state), None)
        if depth is not None and depth <= 0:
            return MinimaxResult(
                self.value(state),
                next(iter(self.actions(state))),
            )
        next_down = None if depth is None else depth - 1

        current_player = self.player(state)
        value: int | float
        if current_player == Player.MAX:
            value = -infinity
            best = max
        else:
            value = infinity
            best = min

        best_action: Action | None = None
        for action in self.actions(state):
            result = self.minimax(self.result(state, action), next_down)
            new_value = best(value, result.value)
            if new_value != value:
                best_action = action
            value = new_value
        return MinimaxResult(value, best_action)


class AsyncMinimax(ABC, Generic[State, Action]):
    """Base class for Minimax AIs."""

    __slots__ = ()

    @classmethod
    @abstractmethod
    async def value(cls, state: State) -> int | float:
        """Return the value of a given game state."""

    @classmethod
    @abstractmethod
    async def terminal(cls, state: State) -> bool:
        """Return if given game state is terminal."""

    @classmethod
    @abstractmethod
    async def player(cls, state: State) -> Player:
        """Return player status given the state of the game.

        Must return either Player.MIN or Player.MAX
        """

    @classmethod
    @abstractmethod
    async def actions(cls, state: State) -> Iterable[Action]:
        """Return a collection of all possible actions in a given game state."""

    @classmethod
    @abstractmethod
    async def result(cls, state: State, action: Action) -> State:
        """Return new game state after performing action on given state."""

    @classmethod
    async def minimax(
        cls,
        state: State,
        depth: int | None = 5,
    ) -> MinimaxResult[Action]:
        """Return minimax result best action for a given state for the current player."""
        if await cls.terminal(state):
            return MinimaxResult(await cls.value(state), None)
        if depth is not None and depth <= 0:
            return MinimaxResult(
                await cls.value(state),
                next(iter(await cls.actions(state))),
            )
        next_down = None if depth is None else depth - 1

        current_player = await cls.player(state)
        value: int | float
        if current_player == Player.MAX:
            value = -infinity
            best = max
        else:
            value = infinity
            best = min

        best_action: Action | None = None
        for action in await cls.actions(state):
            result = await cls.minimax(
                await cls.result(state, action),
                next_down,
            )
            new_value = best(value, result.value)
            if new_value != value:
                best_action = action
            value = new_value
        return MinimaxResult(value, best_action)


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.\n")
