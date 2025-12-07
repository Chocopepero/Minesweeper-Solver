"""
Main script for Minesweeper Solver
Captures the game board, detects cells, and classifies their states
"""

import sys
import cv2
import time
from src.screengrabber import capture_minesweeper_board, find_minesweeper_window
from src.board_detector import BoardDetector
from src.cell_classifier import CellClassifier, CellState
from src.solver import MinesweeperSolver
from src.mouse_controller import MouseController


def count_game_state(board_state):
    """Count different cell types to track game progress."""
    counts = {}
    for row in board_state:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def is_game_over(board_state, rows, cols):
    """Check if game is over (won or lost)."""
    counts = count_game_state(board_state)
    # If there are mines visible, we lost
    if counts.get(CellState.MINE, 0) > 0:
        # Calculate completion percentage
        total_cells = rows * cols
        unrevealed = counts.get(CellState.UNREVEALED, 0)
        revealed = total_cells - unrevealed
        completion = (revealed / total_cells) * 100
        return True, f"LOST ({completion:.1f}% complete)"

    # Check if we won - ONLY when all non-mine cells are revealed
    unrevealed = counts.get(CellState.UNREVEALED, 0)
    flags = counts.get(CellState.FLAG, 0)

    # The only win condition: No unrevealed cells remaining
    # (all non-mine cells have been revealed)
    if unrevealed == 0 and flags > 0:
        return True, "WON"

    # Game is not over if there are still unrevealed cells
    return False, None



def run_solver(rows, cols, board_region):
    """Run the auto-solver loop."""
    print("=" * 60)
    print("Starting Auto-Solver")
    print("=" * 60)

    # Initialize tracking variables
    guess_moves = 0

    # Get window handle
    print("\nFinding Minesweeper window...")
    hwnd = find_minesweeper_window()
    if hwnd is None:
        print("❌ Failed to find window")
        return 1

    # Initialize components
    print("Initializing solver components...")
    detector = BoardDetector()
    solver = MinesweeperSolver(rows, cols)
    classifier = CellClassifier()
    mouse = MouseController(
        hwnd,
        board_x=BoardDetector.DEFAULT_SIDE_OFFSET,
        board_y=BoardDetector.DEFAULT_TOP_OFFSET,
        cell_size=16
    )

    print(f"✅ Ready to solve {rows}×{cols} board\n")

    # Check if board is fresh (all unrevealed)
    board_image = capture_minesweeper_board()
    board = detector.extract_board(board_image, board_region)
    cells = detector.extract_cells(board, rows, cols)
    initial_state = classifier.classify_board(cells)
    counts = count_game_state(initial_state)

    # If board is completely unrevealed, make first click in the center
    if counts.get(CellState.UNREVEALED, 0) == rows * cols:
        center_row = rows // 2
        center_col = cols // 2
        print(f"🎯 Board is fresh. Making first click in the center at ({center_row}, {center_col})...")
        # Click at center of the board
        mouse.click_cell(center_row, center_col)
        time.sleep(0.5)

    # Main solving loop
    print("\nStarting solving loop...")
    print("=" * 60)

    iteration = 0
    max_iterations = 1000
    last_unrevealed_count = -1
    stuck_counter = 0
    last_safe_cells = set()
    last_mines = set()

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

        # Check if game is over FIRST before doing anything else
        game_over, result = is_game_over(board_state, rows, cols)
        if game_over:
            is_won = "WON" in result
            print(f"\n{'🎉' if is_won else '💥'} Game Over: {result}!")
            classifier.print_board_state(board_state)
            break

        # Only proceed with finding moves if game is NOT over
        # Find moves
        solution = solver.solve_step(board_state)
        safe_cells = solution['safe_cells']
        mines = solution['mines']

        # Print what we found
        solver.print_solution(safe_cells, mines)

        # Check if we're stuck (finding the same cells repeatedly)
        if safe_cells == last_safe_cells and mines == last_mines and (safe_cells or mines):
            stuck_counter += 1
            if stuck_counter >= 3:
                print("\n⚠️ Solver is stuck - finding the same cells repeatedly.")
                print("These cells may already be revealed or incorrectly identified.")
                # List the problematic cells for debugging
                if safe_cells:
                    print(f"Stuck on cells: {list(safe_cells)[:5]}")
                break
        else:
            stuck_counter = 0
            last_safe_cells = safe_cells.copy() if safe_cells else set()
            last_mines = mines.copy() if mines else set()

        # If no moves found, make an educated guess
        if not safe_cells and not mines:
            # First check if game is already lost (mine visible)
            counts = count_game_state(board_state)
            if counts.get(CellState.MINE, 0) > 0:
                print("\n💥 Game already lost - mine is visible.")
                break

            guess = solver.make_educated_guess(board_state)
            if guess is None:
                print("\n⚠️  No unrevealed cells found. Game may be complete.")
                classifier.print_board_state(board_state)
                break

            print(f"\n🎲 Making probability-based guess at ({guess[0]}, {guess[1]})")
            safe_cells = {guess}  # Treat guess as a safe cell to click
            guess_moves += 1

        # Before making any moves, double-check the game isn't over
        # (in case we missed a mine from the previous move)
        counts = count_game_state(board_state)
        if counts.get(CellState.MINE, 0) > 0:
            print("\n💥 Mine detected! Game is already over.")
            break

        # Flag mines first
        if mines:
            print(f"\n🚩 Flagging {len(mines)} mines...")
            mouse.click_cells(list(mines), right_click=True, delay=0.01)

        # Click safe cells
        if safe_cells:
            print(f"\n✅ Clicking {len(safe_cells)} safe cells...")
            mouse.click_cells(list(safe_cells), right_click=False, delay=0.01)

            # After clicking, immediately check if we hit a mine
            time.sleep(0.2)  # Small delay to let the board update
            quick_check = capture_minesweeper_board()
            if quick_check is not None:
                quick_board = detector.extract_board(quick_check, board_region)
                quick_cells = detector.extract_cells(quick_board, rows, cols)
                quick_state = classifier.classify_board(quick_cells)
                game_over, result = is_game_over(quick_state, rows, cols)
                if game_over:
                    is_won = "WON" in result
                    print(f"\n{'🎉' if is_won else '💥'} Game Over: {result}!")
                    classifier.print_board_state(quick_state)
                    break

        # Show progress
        counts = count_game_state(board_state)
        unrevealed = counts.get(CellState.UNREVEALED, 0)
        print(f"\n📊 Progress: {unrevealed} cells remaining")

        # Check if we're stuck (unrevealed count not changing)
        if unrevealed == last_unrevealed_count and unrevealed > 0:
            stuck_counter += 1
            if stuck_counter >= 3:
                print("\n⚠️ No progress being made - board state is not changing.")
                print("The game may be complete or the solver is stuck.")
                # Do one final check to see if game is actually won
                final_check = capture_minesweeper_board()
                if final_check is not None:
                    final_board = detector.extract_board(final_check, board_region)
                    final_cells = detector.extract_cells(final_board, rows, cols)
                    final_state = classifier.classify_board(final_cells)
                    game_over, result = is_game_over(final_state, rows, cols)
                    if game_over:
                        print(f"\n{'🎉' if 'WON' in result else '💥'} Game Over: {result}!")
                break
        else:
            last_unrevealed_count = unrevealed

    print("\n" + "=" * 60)
    print("🏁 GAME RESULTS")
    print("=" * 60)
    print(f"🎲 Guess Moves: {guess_moves}")
    print("=" * 60)
    return 0




def main():
    """Run the complete Minesweeper detection pipeline."""

    print("=" * 50)
    print("Minesweeper Solver")
    print("=" * 50)

    # Step 1: Capture the board
    print("\n[1/4] Capturing Minesweeper window...")
    board_image = capture_minesweeper_board()

    if board_image is None:
        print("❌ Failed to capture board. Make sure Minesweeper Arbiter is open.")
        return 1

    print("✅ Screenshot captured")

    # Step 2: Detect board region
    print("\n[2/4] Detecting board region...")
    detector = BoardDetector()
    board_region = detector.detect_board_region(board_image)

    if board_region is None:
        print("❌ Failed to detect board region")
        return 1

    print(f"✅ Board region detected: {board_region}")

    # Step 3: Auto-detect board size and extract cells
    print("\n[3/4] Auto-detecting board size...")
    board = detector.extract_board(board_image, board_region)
    rows, cols = detector.auto_detect_board_size(board)

    print(f"Extracting {rows}×{cols} cells...")
    cells = detector.extract_cells(board, rows, cols)
    print("✅ Cells extracted")

    # Step 4: Classify cells
    print("\n[4/4] Classifying cell states...")
    classifier = CellClassifier()
    board_state = classifier.classify_board(cells)
    print("✅ Classification complete")

    # Display results
    print("\n" + "=" * 50)
    classifier.print_board_state(board_state)
    print("=" * 50)

    # Statistics
    from src.cell_classifier import CellState
    stats = {}
    for row in board_state:
        for cell in row:
            stats[cell] = stats.get(cell, 0) + 1

    print("\n📊 Cell Statistics:")
    for state, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {state.value:12s}: {count:3d} cells")

    # Ask if user wants to auto-solve
    print("\n" + "=" * 50)
    response = input("\nWould you like to auto-solve this board? (y/n): ").strip().lower()

    if response == 'y':
        print("\n🤖 Starting auto-solver...")
        return run_solver(rows, cols, board_region)
    else:
        print("\n✅ Analysis complete. Exiting.")
        return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
