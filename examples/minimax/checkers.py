"""Checkers State."""

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

__title__ = "Checkers State"
__author__ = "CoolCat467"
__license__ = "GNU General Public License Version 3"
__version__ = "0.0.0"

import copy
import math
from collections import Counter
from typing import TYPE_CHECKING, Any, NamedTuple, Self, TypeVar, cast

from minimax import Minimax, MinimaxResult, Player

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

# Note: Tile Ids are chess board tile titles, A1 to H8
# A8 ... H8
# .........
# A1 ... H1

# Player:
# 0 = False = Red   = MIN = 0, 2
# 1 = True  = Black = MAX = 1, 3

T = TypeVar("T")

Pos = tuple[int, int]


class Action(NamedTuple):
    """Represents an action."""

    from_pos: Pos
    to_pos: Pos


class ActionSet(NamedTuple):
    """Represents a set of actions."""

    jumps: dict[Pos, list[Pos]]
    moves: tuple[Pos, ...]
    ends: set[Pos]


def get_sides(xy: Pos) -> tuple[Pos, Pos, Pos, Pos]:
    """Return the tile xy choordinates on the top left, top right, bottom left, and bottom right sides of given xy choordinates."""
    cx, cy = xy
    sides = []
    for raw_dy in range(2):
        dy = raw_dy * 2 - 1
        ny = cy + dy
        for raw_dx in range(2):
            dx = raw_dx * 2 - 1
            nx = cx + dx
            sides.append((nx, ny))
    tuple_sides = tuple(sides)
    assert len(tuple_sides) == 4
    return cast(tuple[Pos, Pos, Pos, Pos], tuple_sides)


def pawn_modify(moves: tuple[T, ...], piece_type: int) -> tuple[T, ...]:
    """Return moves but remove invalid moves for pawns."""
    assert (
        len(moves) == 4
    ), "Tuple size MUST be four for this to return valid results!"
    if (
        piece_type == 0
    ):  # If it's a white pawn, it can only move to top left and top right
        return moves[:2]
    if (
        piece_type == 1
    ):  # If it's a black pawn, it can only move to bottom left anf bottom right
        return moves[2:]
    return moves


class State:
    """Represents state of checkers game."""

    __slots__ = ("size", "turn", "pieces", "pre_calculated_actions")

    def __init__(
        self,
        size: tuple[int, int],
        turn: int,
        pieces: dict[Pos, int],
        /,
        pre_calculated_actions: dict[Pos, ActionSet] | None = None,
    ) -> None:
        """Initialize state."""
        self.size = size
        self.turn = turn
        self.pieces = pieces

        if pre_calculated_actions is None:
            pre_calculated_actions = {}
        self.pre_calculated_actions = pre_calculated_actions

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.size}, {self.turn}, {self.pieces})"

    def __str__(self) -> str:
        """Return text representation of game board state."""
        map_ = {None: "🮀", 0: "🮧", 1: "🮽", 2: "🮮", 3: "🮿"}
        w, h = self.size
        lines = []
        for y in range(h):
            line = []
            for x in range(w):
                if (x + y) % 2:
                    # line.append("_")
                    line.append("🮘")
                    continue
                line.append(map_[self.pieces.get((x, y))])
            lines.append("".join(line))
        ##            lines.append(" | ".join(line))
        ##            lines.append("--+-"*(w-1)+"-")
        return "\n".join(lines)

    @classmethod
    def from_game_board(cls, board_data: dict[str, Any]) -> Self:
        """Return new instance from board data."""
        size = board_data.get("boardsize", (8, 8))
        turn = True
        pieces = cls.get_pieces_from_tiles(board_data.get("tiles", {}))
        return cls(size, turn, pieces)

    @staticmethod
    def get_pieces_from_tiles(
        tiles: dict[str, dict[str, Any]],
    ) -> dict[Pos, int]:
        """Convert board data from game to internal representation."""
        pieces: dict[Pos, int] = {}
        for _tile_name, tile_data in tiles.items():
            piece_type = tile_data["piece"]
            if piece_type in {None, "None"}:
                continue
            x, y = tile_data["xy"]
            pieces[(x, y)] = int(piece_type)
        return pieces

    def calculate_actions(self, position: Pos) -> ActionSet:
        """Return actions the piece at given position can make."""
        jumps = self.get_jumps(position)
        moves = self.get_moves(position)
        ends = set(jumps)
        ends.update(moves)
        return ActionSet(jumps, moves, ends)

    def get_actions_set(self, piece_position: Pos) -> ActionSet:
        """Return potentially cached actions."""
        if piece_position in self.pre_calculated_actions:
            new_action_set = self.pre_calculated_actions[piece_position]
        else:
            new_action_set = self.calculate_actions(piece_position)
            self.pre_calculated_actions[piece_position] = new_action_set
        return new_action_set

    def invalidate_location(self, position: Pos) -> None:
        """Delete pre-calculated actions for a given position if calculated."""
        if position in self.pre_calculated_actions:
            del self.pre_calculated_actions[position]

    def invalidate_all_locations(self) -> None:
        """Clear all pre-calculated actions."""
        self.pre_calculated_actions.clear()

    ##            print(position)

    def piece_kinged(self, piece_pos: Pos, new_type: int) -> None:
        """Piece kinged."""
        ##        print(f'piece_kinged {piece = }')
        self.invalidate_location(piece_pos)

    def piece_moved(self, start_pos: Pos, end_pos: Pos) -> None:
        """Piece moved from start_pos to end_pos."""
        self.invalidate_location(start_pos)

    def piece_jumped(self, jumped_piece_pos: Pos) -> None:
        """Piece has been jumped."""
        ##        print(f'piece_jumped {position = }')
        self.invalidate_all_locations()

    def preform_action(self, action: Action) -> Self:
        """Return new state after performing action on self."""
        from_pos, to_pos = action

        pieces_copy = dict(self.pieces.items())

        # Remove piece from it's start position
        piece_type = pieces_copy.pop(from_pos)

        # See if it's a jump
        if to_pos not in self.get_moves(from_pos):
            # Jumps are more complex to calculate and we need
            # to know what pieces got jumped over
            cur_x, cur_y = from_pos
            for jumped_pos in self.get_jumps(from_pos)[to_pos]:
                from_pos = (cur_x, cur_y)

                # Remove jumped position from pieces in play
                if jumped_pos in pieces_copy:
                    pieces_copy.pop(jumped_pos)
                self.piece_jumped(jumped_pos)
                # See if piece kinged
                jumped_x, jumped_y = jumped_pos
                # Rightshift 1 is more efficiant way to multiply by 2
                cur_x += (jumped_x - cur_x) << 1
                cur_y += (jumped_y - cur_y) << 1

                self.piece_moved(from_pos, (cur_x, cur_y))

                # Now that we know the current position, see if kinged
                if self.does_piece_king(piece_type, (cur_x, cur_y)):
                    piece_type += 2
                    self.piece_kinged((cur_x, cur_y), piece_type)
        else:
            self.piece_moved(from_pos, to_pos)

        # See if it kings and king it if so
        if self.does_piece_king(piece_type, to_pos):
            piece_type += 2
            self.piece_kinged(to_pos, piece_type)

        # Move piece to it's end position
        pieces_copy[to_pos] = piece_type
        self.invalidate_location(from_pos)

        self.invalidate_all_locations()

        # Swap turn
        return self.__class__(
            self.size,
            not self.turn,
            pieces_copy,
            pre_calculated_actions=self.pre_calculated_actions,
        )

    def get_tile_name(self, x: int, y: int) -> str:
        """Return name of a given tile."""
        return chr(65 + x) + str(self.size[1] - y)

    def get_tile_pos(self, name: str) -> Pos:
        """Return tile position from it's name."""
        x = ord(name[0]) - 65
        y = self.size[1] - int(name[1:])
        return (x, y)

    def action_from_points(self, start: Pos, end: Pos) -> Action:
        """Return action from given start and end coordinates."""
        ##        return Action(self.get_tile_name(*start), self.get_tile_name(*end))
        return Action(start, end)

    def get_turn(self) -> int:
        """Return whose turn it is. 0 = red, 1 = black."""
        return self.turn

    def valid_location(self, position: Pos) -> bool:
        """Return if position is valid."""
        x, y = position
        w, h = self.size
        return x >= 0 and y >= 0 and x < w and y < h

    def does_piece_king(self, piece_type: int, position: Pos) -> bool:
        """Return if piece needs to be kinged given it's type and position."""
        _, y = position
        _, h = self.size
        return (piece_type == 0 and y == 0) or (piece_type == 1 and y == h - 1)

    @staticmethod
    def get_enemy(self_type: int) -> int:
        """Return enemy pawn piece type."""
        # If we are kinged, get a pawn version of ourselves.
        # Take that plus one mod 2 to get the pawn of the enemy
        return (self_type + 1) % 2

    @staticmethod
    def get_piece_types(self_type: int) -> tuple[int, int]:
        """Return piece types of given piece type."""
        # If we are kinged, get a pawn version of ourselves.
        self_pawn = self_type % 2
        return (self_pawn, self_pawn + 2)

    def get_jumps(
        self,
        position: Pos,
        piece_type: int | None = None,
        _pieces: dict[Pos, int] | None = None,
        _recursion: int = 0,
    ) -> dict[Pos, list[Pos]]:
        """Return valid jumps a piece can make.

        position is a xy coordinate tuple pointing to a board position
            that may or may not have a piece on it.
        piece_type is the piece type at position. If not
            given, position must point to a tile with a piece on it

        Returns dictionary that maps end positions to
        jumped pieces to get there
        """
        if piece_type is None:
            piece_type = self.pieces[position]
        if _pieces is None:
            _pieces = self.pieces
        _pieces = copy.deepcopy(_pieces)

        enemy_pieces = self.get_piece_types(self.get_enemy(piece_type))

        # Get the side choordinates of the tile and make them tuples so
        # the scan later works properly.
        sides = get_sides(position)
        # Make a dictionary to find what direction a tile is in if you
        # give it the tile.
        # end position : jumped pieces

        # Make a dictionary for the valid jumps and the pieces they jump
        valid: dict[Pos, list[Pos]] = {}

        # For each side tile in the jumpable tiles for this type of piece,
        for direction, side in pawn_modify(
            tuple(enumerate(sides)),
            piece_type,
        ):
            # Make sure side exists
            if not self.valid_location(side):
                continue
            side_piece = _pieces.get(side)
            # Side piece must be one of our enemy's pieces
            if side_piece not in enemy_pieces:
                continue
            # Get the direction from the dictionary we made earlier
            # Get the coordinates of the tile on the side of the main tile's
            # side in the same direction as the main tile's side
            side_side = get_sides(side)[direction]
            # Make sure side exists
            if not self.valid_location(side_side):
                continue
            side_side_piece = _pieces.get(side_side)
            # If the side is open,
            if side_side_piece is None:
                # Add it the valid jumps dictionary and add the tile
                # to the list of end tiles.
                valid[side_side] = [side]

                # Remove jumped piece from future calculations
                _pieces.pop(side)

        # For each end point tile in the list of end point tiles,
        for end_tile in tuple(valid):
            # Get the dictionary from the jumps you could make
            # from that end tile
            w, h = self.size
            if _recursion + 1 > math.ceil((w**2 + h**2) ** 0.25):
                break
            # If the piece has made it to the opposite side,
            piece_type_copy = piece_type
            if self.does_piece_king(piece_type_copy, end_tile):
                # King that piece
                piece_type_copy += 2
                _recursion = -1
            add_valid = self.get_jumps(
                end_tile,
                piece_type_copy,
                _pieces=_pieces,
                _recursion=_recursion + 1,
            )
            # For each key in the new dictionary of valid tile's keys,
            for end_pos, jumped_pieces in add_valid.items():
                # If the key is not already existent in the list of
                # valid destinations,
                if end_pos not in valid:
                    # Add that destination to the dictionary and every
                    # tile you have to jump to get there.
                    no_duplicates = [
                        p for p in jumped_pieces if p not in valid[end_tile]
                    ]
                    valid[end_pos] = valid[end_tile] + no_duplicates

        return valid

    def get_moves(self, position: Pos) -> tuple[Pos, ...]:
        """Return valid moves piece at position can make, not including jumps."""
        piece_type = self.pieces[position]
        # Get the side xy choords of the tile's xy pos,
        # then modify results for pawns
        moves = pawn_modify(get_sides(position), piece_type)
        return tuple(
            [
                m
                for m in filter(self.valid_location, moves)
                if m not in self.pieces
            ],
        )

    def get_actions(self, position: Pos) -> Generator[Action, None, None]:
        """Yield all moves and jumps the piece at position can make."""
        ends = set(self.get_jumps(position))
        ends.update(self.get_moves(position))
        ##        ends = self.get_actions_set(position).ends
        for end in ends:
            yield self.action_from_points(position, end)

    def get_all_actions(self, player: int) -> Generator[Action, None, None]:
        """Yield all actions for given player."""
        player_pieces = {player, player + 2}
        for position, piece_type in self.pieces.items():
            if piece_type not in player_pieces:
                continue
            yield from self.get_actions(position)

    def check_for_win(self) -> int | None:
        """Return player number if they won else None."""
        # For each of the two players,
        for player in range(2):
            # For each tile in the playable tiles,
            has_move = False
            for _ in self.get_all_actions(player):
                has_move = True
                # Player has at least one move, no need to continue
                break
            if not has_move and self.turn == player:
                # Continued without break, so player either has no moves
                # or no possible moves, so their opponent wins
                return (player + 1) % 2
        return None

    def can_player_select_piece(self, player: int, tile_pos: Pos) -> bool:
        """Return True if player can select piece on given tile position."""
        piece_at_pos = self.pieces.get(tile_pos)
        if piece_at_pos is None:
            return False
        return (piece_at_pos % 2) == player

    def get_pieces(self) -> tuple[tuple[Pos, int], ...]:
        """Return all pieces."""
        return tuple((pos, type_) for pos, type_ in self.pieces.items())


class CheckersMinimax(Minimax[State, Action]):
    """Minimax Algorithm for Checkers."""

    __slots__ = ()

    def value(self, state: State) -> int | float:
        """Return value of given game state."""
        # Return winner if possible
        win = state.check_for_win()
        # If no winner, we have to predict the value
        if win is None:
            # We'll estimate the value by the pieces in play
            counts = Counter(state.pieces.values())
            # Score is pawns plus 3 times kings
            min_ = counts[0] + 3 * counts[2]
            max_ = counts[1] + 3 * counts[3]
            # More max will make score higher,
            # more min will make score lower
            # Plus one in divisor makes so never / 0
            return (max_ - min_) / (max_ + min_ + 1)
        return win * 2 - 1

    @staticmethod
    def terminal(state: State) -> bool:
        """Return if game state is terminal."""
        return state.check_for_win() is not None

    @staticmethod
    def player(state: State) -> Player:
        """Return Player enum from current state's turn."""
        return Player.MAX if state.get_turn() else Player.MIN

    @staticmethod
    def actions(state: State) -> Iterable[Action]:
        """Return all actions that are able to be performed for the current player in the given state."""
        return state.get_all_actions(int(state.get_turn()))

    @staticmethod
    def result(state: State, action: Action) -> State:
        """Return new state after performing given action on given current state."""
        return state.preform_action(action)

    def adaptive_depth_minimax(
        self,
        state: State,
        minimum: int,
        maximum: int,
    ) -> MinimaxResult[Action]:
        """Return minimax result from adaptive max depth."""
        ##        types = state.pieces.values()
        ##        current = len(types)
        ##        w, h = state.size
        ##        max_count = w * h // 6 << 1
        ##        old_depth = (1 - (current / max_count)) * math.floor(
        ##            math.sqrt(w**2 + h**2)
        ##        )

        depth = self.value(state) * maximum + minimum
        final_depth = min(maximum, max(minimum, math.floor(depth)))
        print(f"{depth = } {final_depth = }")
        return self.minimax(state, final_depth)


def generate_pieces(
    board_width: int,
    board_height: int,
    colors: int = 2,
) -> dict[Pos, int]:
    """Generate data about each piece."""
    pieces: dict[Pos, int] = {}
    # Get where pieces should be placed
    z_to_1 = round(board_height / 3)  # White
    z_to_2 = (board_height - (z_to_1 * 2)) + z_to_1  # Black
    # For each xy position in the area of where tiles should be,
    for y in range(board_height):
        # Reset the x pos to 0
        for x in range(board_width):
            # Get the color of that spot by adding x and y mod the number of different colors
            color = (x + y) % colors
            # If a piece should be placed on that tile and the tile is not Red,
            if (not color) and ((y <= z_to_1 - 1) or (y >= z_to_2)):
                # Set the piece to White Pawn or Black Pawn depending on the current y pos
                piece_type = int(y <= z_to_1)
                pieces[(x, y)] = piece_type
    return pieces


def run() -> None:
    """Run test of module."""
    size = (8, 8)
    state = State(size, 0, generate_pieces(*size))
    ##    print(f'{state!r}')
    ##    print(state)
    # print(f'{tuple(state.valid_moves()) = }')

    minimax = CheckersMinimax()
    max_depth = 4
    while not minimax.terminal(state):
        value, action = minimax.minimax(state, max_depth)
        assert action is not None
        # print('\n'.join(f'{idx}: {act}' for idx, act in enumerate(minimax.actions(state))))
        # print(f'>{action}')
        print(action)
        state = minimax.result(state, action)
        print(state)
        print(repr(state))
        print(minimax.value(state))
        print()


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.\n")
    run()
