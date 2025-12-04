"""
Minesweeper Solver - Uses logic rules to solve the board
"""

from typing import List, Tuple, Set
from src.cell_classifier import CellState

import time
import csv
from datetime import datetime
import os

def log_game_result(game_number, difficulty, status, time_taken, completion,
                    flags_placed=0, guesses_made=0, moves_total=0):
    """
    Logs a single game result to results.csv for data analysis.
    Adds additional fields for flags, guesses, and total moves.
    """
    file_exists = os.path.isfile("results.csv")

    with open("results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "game_number", "difficulty", "status", "time_taken", "completion",
                "flags_placed", "guesses_made", "moves_total"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            game_number,
            difficulty,
            status,
            round(time_taken, 2),
            f"{completion:.1f}%",
            flags_placed,
            guesses_made,
            moves_total
        ])


class MinesweeperSolver:
    """Solves Minesweeper using logical deduction."""

    def __init__(self, rows: int, cols: int):
        """
        Initialize the solver.

        Args:
            rows: Number of rows in the board
            cols: Number of columns in the board
        """
        self.rows = rows
        self.cols = cols

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Get all neighbor coordinates for a cell.

        Args:
            row: Row index
            col: Column index

        Returns:
            List of (row, col) tuples for neighbors
        """
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def count_neighbor_states(self, board_state: List[List[CellState]],
                            row: int, col: int) -> dict:
        """
        Count the different states of neighboring cells.

        Args:
            board_state: 2D list of CellState values
            row: Row index
            col: Column index

        Returns:
            Dict with counts: {'unrevealed': X, 'flags': Y, 'total': Z}
        """
        neighbors = self.get_neighbors(row, col)
        unrevealed = []
        flags = []

        for nr, nc in neighbors:
            state = board_state[nr][nc]
            if state == CellState.UNREVEALED:
                unrevealed.append((nr, nc))
            elif state == CellState.FLAG:
                flags.append((nr, nc))

        return {
            'unrevealed': unrevealed,
            'flags': flags,
            'total': len(neighbors)
        }

    def find_safe_cells(self, board_state: List[List[CellState]]) -> Set[Tuple[int, int]]:
        """
        Find cells that are safe to click using Rule 1:
        If a number cell has exactly that many flags around it,
        all other unrevealed neighbors are safe.

        Args:
            board_state: 2D list of CellState values

        Returns:
            Set of (row, col) tuples that are safe to click
        """
        safe_cells = set()

        # Check each revealed number cell
        for row in range(self.rows):
            for col in range(self.cols):
                state = board_state[row][col]

                # Check if it's a number cell (1-8)
                if state in [CellState.REVEALED_1, CellState.REVEALED_2,
                           CellState.REVEALED_3, CellState.REVEALED_4,
                           CellState.REVEALED_5, CellState.REVEALED_6,
                           CellState.REVEALED_7, CellState.REVEALED_8]:

                    # Get the number value
                    number = int(state.value)

                    # Count neighbors
                    neighbor_info = self.count_neighbor_states(board_state, row, col)

                    # Rule 1: If flags == number, unrevealed cells are safe
                    if len(neighbor_info['flags']) == number:
                        safe_cells.update(neighbor_info['unrevealed'])

        return safe_cells

    def find_mines(self, board_state: List[List[CellState]]) -> Set[Tuple[int, int]]:
        """
        Find cells that must be mines using Rule 2:
        If a number cell has (unrevealed + flags) == number,
        all unrevealed neighbors are mines.

        Args:
            board_state: 2D list of CellState values

        Returns:
            Set of (row, col) tuples that are mines (should be flagged)
        """
        mines = set()

        # Check each revealed number cell
        for row in range(self.rows):
            for col in range(self.cols):
                state = board_state[row][col]

                # Check if it's a number cell (1-8)
                if state in [CellState.REVEALED_1, CellState.REVEALED_2,
                           CellState.REVEALED_3, CellState.REVEALED_4,
                           CellState.REVEALED_5, CellState.REVEALED_6,
                           CellState.REVEALED_7, CellState.REVEALED_8]:

                    # Get the number value
                    number = int(state.value)

                    # Count neighbors
                    neighbor_info = self.count_neighbor_states(board_state, row, col)

                    # Rule 2: If (unrevealed + flags) == number, unrevealed are mines
                    if len(neighbor_info['unrevealed']) + len(neighbor_info['flags']) == number:
                        mines.update(neighbor_info['unrevealed'])

        return mines

    def find_121_pattern(self, board_state: List[List[CellState]]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Find 1-2-1 pattern: When you have 1-2-1 in a row/column,
        the cells on the sides of the 2 are safe.

        Pattern: [1] [2] [1]
                  U  UU  U
        The two U cells above the 2 are mines, the outer U cells are safe.

        Returns:
            (safe_cells, mines) tuple
        """
        safe_cells = set()
        mines = set()

        # Check horizontal 1-2-1 patterns
        for row in range(self.rows):
            for col in range(self.cols - 2):
                # Get three consecutive cells
                c1 = board_state[row][col]
                c2 = board_state[row][col + 1]
                c3 = board_state[row][col + 2]

                # Check if it's 1-2-1
                if c1 == CellState.REVEALED_1 and c2 == CellState.REVEALED_2 and c3 == CellState.REVEALED_1:
                    # Check if pattern is valid (cells above/below are unrevealed)
                    if row > 0:  # Pattern with cells above
                        above = [board_state[row - 1][col], board_state[row - 1][col + 1], board_state[row - 1][col + 2]]
                        if all(s == CellState.UNREVEALED for s in above):
                            # Middle cell above 2 is a mine
                            mines.add((row - 1, col + 1))
                            # Outer cells are safe
                            safe_cells.add((row - 1, col))
                            safe_cells.add((row - 1, col + 2))

                    if row < self.rows - 1:  # Pattern with cells below
                        below = [board_state[row + 1][col], board_state[row + 1][col + 1], board_state[row + 1][col + 2]]
                        if all(s == CellState.UNREVEALED for s in below):
                            # Middle cell below 2 is a mine
                            mines.add((row + 1, col + 1))
                            # Outer cells are safe
                            safe_cells.add((row + 1, col))
                            safe_cells.add((row + 1, col + 2))

        # Check vertical 1-2-1 patterns
        for row in range(self.rows - 2):
            for col in range(self.cols):
                # Get three consecutive cells
                c1 = board_state[row][col]
                c2 = board_state[row + 1][col]
                c3 = board_state[row + 2][col]

                # Check if it's 1-2-1
                if c1 == CellState.REVEALED_1 and c2 == CellState.REVEALED_2 and c3 == CellState.REVEALED_1:
                    # Check if pattern is valid (cells to left/right are unrevealed)
                    if col > 0:  # Pattern with cells to left
                        left = [board_state[row][col - 1], board_state[row + 1][col - 1], board_state[row + 2][col - 1]]
                        if all(s == CellState.UNREVEALED for s in left):
                            # Middle cell left of 2 is a mine
                            mines.add((row + 1, col - 1))
                            # Outer cells are safe
                            safe_cells.add((row, col - 1))
                            safe_cells.add((row + 2, col - 1))

                    if col < self.cols - 1:  # Pattern with cells to right
                        right = [board_state[row][col + 1], board_state[row + 1][col + 1], board_state[row + 2][col + 1]]
                        if all(s == CellState.UNREVEALED for s in right):
                            # Middle cell right of 2 is a mine
                            mines.add((row + 1, col + 1))
                            # Outer cells are safe
                            safe_cells.add((row, col + 1))
                            safe_cells.add((row + 2, col + 1))

        return safe_cells, mines

    def find_1221_pattern(self, board_state: List[List[CellState]]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Find 1-2-2-1 pattern: When you have 1-2-2-1 in a row/column,
        the cells at the ends are safe.

        Pattern: [1] [2] [2] [1]
                  U  UU  UU  U
        The four U cells above the 2-2 are mines, the outer U cells are safe.

        Returns:
            (safe_cells, mines) tuple
        """
        safe_cells = set()
        mines = set()

        # Check horizontal 1-2-2-1 patterns
        for row in range(self.rows):
            for col in range(self.cols - 3):
                # Get four consecutive cells
                c1 = board_state[row][col]
                c2 = board_state[row][col + 1]
                c3 = board_state[row][col + 2]
                c4 = board_state[row][col + 3]

                # Check if it's 1-2-2-1
                if c1 == CellState.REVEALED_1 and c2 == CellState.REVEALED_2 and c3 == CellState.REVEALED_2 and c4 == CellState.REVEALED_1:
                    # Check if pattern is valid (cells above/below are unrevealed)
                    if row > 0:  # Pattern with cells above
                        above = [board_state[row - 1][col], board_state[row - 1][col + 1],
                                board_state[row - 1][col + 2], board_state[row - 1][col + 3]]
                        if all(s == CellState.UNREVEALED for s in above):
                            # Two middle cells are mines
                            mines.add((row - 1, col + 1))
                            mines.add((row - 1, col + 2))
                            # Outer cells are safe
                            safe_cells.add((row - 1, col))
                            safe_cells.add((row - 1, col + 3))

                    if row < self.rows - 1:  # Pattern with cells below
                        below = [board_state[row + 1][col], board_state[row + 1][col + 1],
                                board_state[row + 1][col + 2], board_state[row + 1][col + 3]]
                        if all(s == CellState.UNREVEALED for s in below):
                            # Two middle cells are mines
                            mines.add((row + 1, col + 1))
                            mines.add((row + 1, col + 2))
                            # Outer cells are safe
                            safe_cells.add((row + 1, col))
                            safe_cells.add((row + 1, col + 3))

        # Check vertical 1-2-2-1 patterns
        for row in range(self.rows - 3):
            for col in range(self.cols):
                # Get four consecutive cells
                c1 = board_state[row][col]
                c2 = board_state[row + 1][col]
                c3 = board_state[row + 2][col]
                c4 = board_state[row + 3][col]

                # Check if it's 1-2-2-1
                if c1 == CellState.REVEALED_1 and c2 == CellState.REVEALED_2 and c3 == CellState.REVEALED_2 and c4 == CellState.REVEALED_1:
                    # Check if pattern is valid (cells to left/right are unrevealed)
                    if col > 0:  # Pattern with cells to left
                        left = [board_state[row][col - 1], board_state[row + 1][col - 1],
                               board_state[row + 2][col - 1], board_state[row + 3][col - 1]]
                        if all(s == CellState.UNREVEALED for s in left):
                            # Two middle cells are mines
                            mines.add((row + 1, col - 1))
                            mines.add((row + 2, col - 1))
                            # Outer cells are safe
                            safe_cells.add((row, col - 1))
                            safe_cells.add((row + 3, col - 1))

                    if col < self.cols - 1:  # Pattern with cells to right
                        right = [board_state[row][col + 1], board_state[row + 1][col + 1],
                                board_state[row + 2][col + 1], board_state[row + 3][col + 1]]
                        if all(s == CellState.UNREVEALED for s in right):
                            # Two middle cells are mines
                            mines.add((row + 1, col + 1))
                            mines.add((row + 2, col + 1))
                            # Outer cells are safe
                            safe_cells.add((row, col + 1))
                            safe_cells.add((row + 3, col + 1))

        return safe_cells, mines

    def find_corner_1_pattern(self, board_state: List[List[CellState]]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Find corner 1 pattern: A 1 in the corner means only one mine among its neighbors.

        Pattern (top-left corner):
        [1] U
         U  U

        If the 1 is in a corner, it has only 3 neighbors. We can deduce mines more easily.

        Returns:
            (safe_cells, mines) tuple
        """
        safe_cells = set()
        mines = set()

        # Top-left corner
        if (self.rows > 1 and self.cols > 1 and
            board_state[0][0] == CellState.REVEALED_1):
            neighbors = [(0, 1), (1, 0), (1, 1)]
            unrevealed = [n for n in neighbors if board_state[n[0]][n[1]] == CellState.UNREVEALED]

            # If only one unrevealed neighbor, it's the mine
            if len(unrevealed) == 1:
                mines.add(unrevealed[0])

        # Top-right corner
        if (self.rows > 1 and self.cols > 1 and
            board_state[0][self.cols - 1] == CellState.REVEALED_1):
            neighbors = [(0, self.cols - 2), (1, self.cols - 2), (1, self.cols - 1)]
            unrevealed = [n for n in neighbors if board_state[n[0]][n[1]] == CellState.UNREVEALED]

            if len(unrevealed) == 1:
                mines.add(unrevealed[0])

        # Bottom-left corner
        if (self.rows > 1 and self.cols > 1 and
            board_state[self.rows - 1][0] == CellState.REVEALED_1):
            neighbors = [(self.rows - 2, 0), (self.rows - 2, 1), (self.rows - 1, 1)]
            unrevealed = [n for n in neighbors if board_state[n[0]][n[1]] == CellState.UNREVEALED]

            if len(unrevealed) == 1:
                mines.add(unrevealed[0])

        # Bottom-right corner
        if (self.rows > 1 and self.cols > 1 and
            board_state[self.rows - 1][self.cols - 1] == CellState.REVEALED_1):
            neighbors = [(self.rows - 2, self.cols - 2), (self.rows - 2, self.cols - 1),
                        (self.rows - 1, self.cols - 2)]
            unrevealed = [n for n in neighbors if board_state[n[0]][n[1]] == CellState.UNREVEALED]

            if len(unrevealed) == 1:
                mines.add(unrevealed[0])

        return safe_cells, mines

    def find_subset_overlap_pattern(self, board_state: List[List[CellState]]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Find subset overlap pattern between two number cells.

        Rule: If N(A) ⊆ N(B) (A's unrevealed neighbors are a subset of B's),
        then the mines in N(B) \ N(A) = number(B) - number(A)

        Example:
        [1][2]  where N(1) ⊆ N(2)
         U UU

        If N(1) = {U1} and N(2) = {U1, U2, U3}, then:
        - Mines in N(1) = 1
        - Mines in N(2) = 2
        - Therefore mines in {U2, U3} = 2 - 1 = 1

        If |N(B) \ N(A)| = number(B) - number(A):
            All cells in N(B) \ N(A) are mines
        If number(B) - number(A) = 0:
            All cells in N(B) \ N(A) are safe

        Returns:
            (safe_cells, mines) tuple
        """
        safe_cells = set()
        mines = set()

        # Get all number cells
        number_cells = []
        for row in range(self.rows):
            for col in range(self.cols):
                state = board_state[row][col]
                if state in [CellState.REVEALED_1, CellState.REVEALED_2,
                           CellState.REVEALED_3, CellState.REVEALED_4,
                           CellState.REVEALED_5, CellState.REVEALED_6,
                           CellState.REVEALED_7, CellState.REVEALED_8]:
                    number_cells.append((row, col, int(state.value)))

        # Check all pairs of number cells
        for i in range(len(number_cells)):
            for j in range(i + 1, len(number_cells)):
                row_a, col_a, num_a = number_cells[i]
                row_b, col_b, num_b = number_cells[j]

                # Get unrevealed neighbors (excluding flagged cells)
                neighbors_a = set(self.get_neighbors(row_a, col_a))
                neighbors_b = set(self.get_neighbors(row_b, col_b))

                unrevealed_a = {n for n in neighbors_a if board_state[n[0]][n[1]] == CellState.UNREVEALED}
                unrevealed_b = {n for n in neighbors_b if board_state[n[0]][n[1]] == CellState.UNREVEALED}

                # Count flags already placed
                flags_a = sum(1 for n in neighbors_a if board_state[n[0]][n[1]] == CellState.FLAG)
                flags_b = sum(1 for n in neighbors_b if board_state[n[0]][n[1]] == CellState.FLAG)

                # Remaining mines to find
                remaining_a = num_a - flags_a
                remaining_b = num_b - flags_b

                # Check if A is subset of B
                if unrevealed_a.issubset(unrevealed_b) and unrevealed_a != unrevealed_b:
                    diff = unrevealed_b - unrevealed_a
                    mine_diff = remaining_b - remaining_a

                    if mine_diff == len(diff) and len(diff) > 0:
                        # All cells in diff are mines
                        mines.update(diff)
                    elif mine_diff == 0 and len(diff) > 0:
                        # All cells in diff are safe
                        safe_cells.update(diff)

                # Check if B is subset of A
                elif unrevealed_b.issubset(unrevealed_a) and unrevealed_b != unrevealed_a:
                    diff = unrevealed_a - unrevealed_b
                    mine_diff = remaining_a - remaining_b

                    if mine_diff == len(diff) and len(diff) > 0:
                        # All cells in diff are mines
                        mines.update(diff)
                    elif mine_diff == 0 and len(diff) > 0:
                        # All cells in diff are safe
                        safe_cells.update(diff)

        return safe_cells, mines

    def solve_step(self, board_state: List[List[CellState]]) -> dict:
        """
        Perform one step of solving.

        Args:
            board_state: 2D list of CellState values

        Returns:
            Dict with 'safe_cells' and 'mines' sets
        """
        # Apply basic rules
        safe_cells = self.find_safe_cells(board_state)
        mines = self.find_mines(board_state)

        # If basic rules didn't find anything, try advanced patterns
        if not safe_cells and not mines:
            # Try corner 1 pattern (fast and simple)
            pattern_safe, pattern_mines = self.find_corner_1_pattern(board_state)
            safe_cells.update(pattern_safe)
            mines.update(pattern_mines)

        if not safe_cells and not mines:
            # Try subset overlap (works for any number pair, very powerful)
            pattern_safe, pattern_mines = self.find_subset_overlap_pattern(board_state)
            safe_cells.update(pattern_safe)
            mines.update(pattern_mines)

        if not safe_cells and not mines:
            # Try 1-2-1 pattern
            pattern_safe, pattern_mines = self.find_121_pattern(board_state)
            safe_cells.update(pattern_safe)
            mines.update(pattern_mines)

        if not safe_cells and not mines:
            # Try 1-2-2-1 pattern
            pattern_safe, pattern_mines = self.find_1221_pattern(board_state)
            safe_cells.update(pattern_safe)
            mines.update(pattern_mines)

        return {
            'safe_cells': safe_cells,
            'mines': mines
        }

    def make_educated_guess(self, board_state: List[List[CellState]]) -> Tuple[int, int]:
        """
        Make an educated guess when stuck.
        Strategy: Pick unrevealed cell with most revealed neighbors (safest guess).

        Args:
            board_state: 2D list of CellState values

        Returns:
            (row, col) tuple of cell to guess, or None if no unrevealed cells
        """
        best_cell = None
        max_revealed_neighbors = -1

        for row in range(self.rows):
            for col in range(self.cols):
                if board_state[row][col] != CellState.UNREVEALED:
                    continue

                # Count revealed neighbors
                neighbors = self.get_neighbors(row, col)
                revealed_count = 0

                for nr, nc in neighbors:
                    state = board_state[nr][nc]
                    # Count revealed cells (numbers or empty)
                    if state in [CellState.REVEALED_0, CellState.REVEALED_1,
                               CellState.REVEALED_2, CellState.REVEALED_3,
                               CellState.REVEALED_4, CellState.REVEALED_5,
                               CellState.REVEALED_6, CellState.REVEALED_7,
                               CellState.REVEALED_8]:
                        revealed_count += 1

                # Pick cell with most revealed neighbors (statistically safer)
                if revealed_count > max_revealed_neighbors:
                    max_revealed_neighbors = revealed_count
                    best_cell = (row, col)

        return best_cell

    def print_solution(self, safe_cells: Set[Tuple[int, int]],
                      mines: Set[Tuple[int, int]]):
        """Print the found moves in a readable format."""
        if safe_cells:
            print(f"\n✅ Found {len(safe_cells)} safe cells to click:")
            for row, col in sorted(safe_cells)[:10]:  # Show first 10
                print(f"   - Row {row}, Col {col}")
            if len(safe_cells) > 10:
                print(f"   ... and {len(safe_cells) - 10} more")

        if mines:
            print(f"\n🚩 Found {len(mines)} mines to flag:")
            for row, col in sorted(mines)[:10]:  # Show first 10
                print(f"   - Row {row}, Col {col}")
            if len(mines) > 10:
                print(f"   ... and {len(mines) - 10} more")

        if not safe_cells and not mines:
            print("\n❓ No obvious moves found. Will make an educated guess...")


if __name__ == "__main__":
    print("Minesweeper Solver Module")
    print("Import this module to solve Minesweeper boards.")

    from solver import MinesweeperSolver  # or adjust if your solver class is named differently

    # Initialize game and solver
    solver = MinesweeperSolver()

    # Example: record start time
    start_time = time.time()

    # Run the solver
    solved = True   #placeholder for now
    # Measure total time
    time_elapsed = time.time() - start_time

    # Example placeholder values
    game_id = 1  # later you can auto-increment this
    difficulty = "beginner"
    status = "win" if solved else "fail"
    board_completion = 100.0 if solved else 75.0  # fake completion %

    # Log the result
    log_game_result(
        game_number=game_id,
        difficulty=difficulty,
        status=status,
        time_taken=time_elapsed,
        completion=board_completion
    )

    print(f"Game logged: #{game_id} - {status} in {time_elapsed:.2f}s")
