"""
Auto-Solve Minesweeper - Main solving loop
"""

import sys
import time
from src.screengrabber import capture_minesweeper_board, find_minesweeper_window
from src.board_detector import BoardDetector
from src.cell_classifier import CellClassifier, CellState
from src.solver import MinesweeperSolver
from src.mouse_controller import MouseController


def count_game_state(board_state):
    counts = {}
    for row in board_state:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def is_game_over(board_state):
    """Check if game is over (won or lost)."""
    counts = count_game_state(board_state)
    # If there are mines visible, we lost
    if counts.get(CellState.MINE, 0) > 0:
        return True, "LOST"
    # If only unrevealed cells and flags remain, check if we won
    unrevealed = counts.get(CellState.UNREVEALED, 0)
    flags = counts.get(CellState.FLAG, 0)
    if unrevealed + flags < 10: 
        return True, "WON"
    return False, None


def main():
    """Main auto-solving loop."""
    print("=" * 60)
    print("Minesweeper Auto-Solver")
    print("=" * 60)
    print("\nThis will automatically solve the Minesweeper board.")
    print("Make sure Minesweeper Arbiter is open and visible.")
    print("\nPress Ctrl+C at any time to stop.\n")

    input("Press ENTER to start solving...")

    # Step 1: Get window handle
    print("\n[1/5] Finding Minesweeper window...")
    hwnd = find_minesweeper_window()
    if hwnd is None:
        print("❌ Failed to find window")
        return 1

    # Step 2: Capture initial board
    print("\n[2/5] Capturing initial board...")
    board_image = capture_minesweeper_board()
    if board_image is None:
        print("❌ Failed to capture board")
        return 1

    # Step 3: Detect board dimensions
    print("\n[3/5] Detecting board dimensions...")
    detector = BoardDetector()
    board_region = detector.detect_board_region(board_image)
    if board_region is None:
        print("❌ Failed to detect board region")
        return 1

    board = detector.extract_board(board_image, board_region)
    rows, cols = detector.auto_detect_board_size(board)

    # Step 4: Initialize components
    print("\n[4/5] Initializing solver...")
    solver = MinesweeperSolver(rows, cols)
    classifier = CellClassifier()
    mouse = MouseController(
        hwnd,
        board_x=BoardDetector.DEFAULT_SIDE_OFFSET,
        board_y=BoardDetector.DEFAULT_TOP_OFFSET,
        cell_size=16
    )

    print(f"✅ Ready to solve {rows}×{cols} board")

    # Step 5: Check if board is fresh (all unrevealed)
    cells = detector.extract_cells(board, rows, cols)
    initial_state = classifier.classify_board(cells)
    counts = count_game_state(initial_state)

    print(f"\nInitial board state:")
    print(f"  Unrevealed: {counts.get(CellState.UNREVEALED, 0)}")
    print(f"  Revealed: {rows * cols - counts.get(CellState.UNREVEALED, 0)}")

    if counts.get(CellState.UNREVEALED, 0) == rows * cols:
        print("\n🎯 Board is fresh. Making first click in the center...")
        center_row, center_col = rows // 2, cols // 2
        mouse.click_cell(center_row, center_col)
        time.sleep(0.5)  

    # Step 6: Main solving loop
    print("\n[5/5] Starting solving loop...")
    print("=" * 60)

    iteration = 0
    max_iterations = 1000  # Safety limit

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Capture current board state
        board_image = capture_minesweeper_board()
        if board_image is None:
            print("❌ Failed to capture board")
            break

        board = detector.extract_board(board_image, board_region)
        cells = detector.extract_cells(board, rows, cols)
        board_state = classifier.classify_board(cells)

        # Check if game is over
        game_over, result = is_game_over(board_state)
        if game_over:
            print(f"\n{'🎉' if result == 'WON' else '💥'} Game Over: {result}!")
            classifier.print_board_state(board_state)
            break

        # Find moves
        solution = solver.solve_step(board_state)
        safe_cells = solution['safe_cells']
        mines = solution['mines']

        # Print what we found
        solver.print_solution(safe_cells, mines)

        # If no moves found, we're stuck
        if not safe_cells and not mines:
            print("\n⚠️  Solver is stuck. No obvious moves.")
            print("You may need to make a guess or use advanced techniques.")
            classifier.print_board_state(board_state)
            break

        # Flag mines first
        if mines:
            print(f"\n🚩 Flagging {len(mines)} mines...")
            mouse.click_cells(list(mines), right_click=True, delay=0.1)
            time.sleep(0.3)

        # Click safe cells
        if safe_cells:
            print(f"\n✅ Clicking {len(safe_cells)} safe cells...")
            mouse.click_cells(list(safe_cells), right_click=False, delay=0.1)
            time.sleep(0.3)

        # Show progress
        counts = count_game_state(board_state)
        unrevealed = counts.get(CellState.UNREVEALED, 0)
        print(f"\n📊 Progress: {unrevealed} cells remaining")

    print("\n" + "=" * 60)
    print("Solving complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
