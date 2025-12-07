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

from src.solver import log_game_result 
import time



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
    return False, None


def run_solver(rows, cols, board_region):
    """Run the auto-solver loop."""
    print("=" * 60)
    print("Starting Auto-Solver")
    print("=" * 60)

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

    #initialize performance counters
    flags_placed = 0
    guesses_made = 0
    moves_total = 0
    start_time = time.time()

    # Check if board is fresh (all unrevealed)
    board_image = capture_minesweeper_board()
    board = detector.extract_board(board_image, board_region)
    cells = detector.extract_cells(board, rows, cols)
    initial_state = classifier.classify_board(cells)
    counts = count_game_state(initial_state)

    # If board is completely unrevealed, make first click in top-left corner
    if counts.get(CellState.UNREVEALED, 0) == rows * cols:
        print("🎯 Board is fresh. Making first click in the top-left corner...")
        # Click at (0, 0) - top-left corner
        mouse.click_cell(0, 0)
        time.sleep(0.5)

    # Main solving loop
    print("\nStarting solving loop...")
    print("=" * 60)

    iteration = 0
    max_iterations = 1000

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
        game_over, result = is_game_over(board_state, rows, cols)
        if game_over:
            is_won = "WON" in result
            print(f"\n{'🎉' if is_won else '💥'} Game Over: {result}!")
            classifier.print_board_state(board_state)

            # === LOGGING SECTION ===
            time_elapsed = time.time() - start_time

            # Detect difficulty
            if rows == 9 and cols == 9:
                difficulty = "beginner"
            elif rows == 16 and cols == 16:
                difficulty = "intermediate"
            elif rows == 16 and cols == 30:
                difficulty = "expert"
            else:
                difficulty = f"{rows}x{cols}"

            # Determine status and completion %
            status = "win" if is_won else "fail"
            import re
            match = re.search(r"\((\d+\.\d+)% complete", result)
            completion = float(match.group(1)) if match else (100.0 if is_won else 0.0)

            # Auto-generate game number
            import os
            game_id = 1
            if os.path.exists("results.csv"):
                with open("results.csv") as f:
                    game_id = sum(1 for _ in f)

            # === POST-GAME REPORT ===
            print("\n📋 Game Summary Report")
            print("───────────────────────────────")
            print(f"Completion:       {'Yes' if status == 'win' else 'No'}")
            print(f"Time Elapsed:     {time_elapsed:.2f} seconds")
            print(f"# of Moves:       {moves_total}")
            print(f"Correct Flags:    {flags_placed}")
            print(f"Guess Moves:      {guesses_made}")
            print("───────────────────────────────")

            # Log the result
            log_game_result(
                game_number=game_id,
                difficulty=difficulty,
                status=status,
                time_taken=time_elapsed,
                completion=completion,
                flags_placed=flags_placed,
                guesses_made=guesses_made,
                moves_total=moves_total
            )

            print(f"\n✅ Logged Game #{game_id}: {difficulty.title()} - {status.upper()} ({completion:.1f}%, {time_elapsed:.2f}s)")
            break

        # Find moves
        solution = solver.solve_step(board_state)
        safe_cells = solution['safe_cells']
        mines = solution['mines']

        # Print what we found
        solver.print_solution(safe_cells, mines)

        # If no moves found, make an educated guess
        if not safe_cells and not mines:
            guess = solver.make_educated_guess(board_state)
            if guess is None:
                print("\n⚠️  No unrevealed cells found. Game may be complete.")
                classifier.print_board_state(board_state)
                break

            print(f"\n🎲 Making educated guess at ({guess[0]}, {guess[1]})")
            safe_cells = {guess}
            guesses_made += 1  # count each educated guess as a guess move

        # Flag mines first 
        if mines:
            print(f"\n🚩 Flagging {len(mines)} mines...")
            mouse.click_cells(list(mines), right_click=True, delay=0.01)
            flags_placed += len(mines)
            moves_total += len(mines)

        # Click safe cells
        if safe_cells:
            print(f"\n✅ Clicking {len(safe_cells)} safe cells...")
            mouse.click_cells(list(safe_cells), right_click=False, delay=0.01)
            moves_total += len(safe_cells)

        # Show progress
        counts = count_game_state(board_state)
        unrevealed = counts.get(CellState.UNREVEALED, 0)
        print(f"\n📊 Progress: {unrevealed} cells remaining")

    print("\n" + "=" * 60)
    print("Solving complete!")
    print(f"🎲 Total guess moves made: {guesses_made}")
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

def wait_for_new_game(detector, classifier, rows, cols, timeout=60):
    """Wait until a fresh unrevealed board appears (new game)."""
    start = time.time()
    print("\n⌛ Waiting for new game to start...")

    while time.time() - start < timeout:
        board_image = capture_minesweeper_board()
        board_region = detector.detect_board_region(board_image)
        board = detector.extract_board(board_image, board_region)
        cells = detector.extract_cells(board, rows, cols)
        board_state = classifier.classify_board(cells)

        # Check if every cell is still unrevealed (fresh board)
        if all(cell == CellState.UNREVEALED for row in board_state for cell in row):
            print("🆕 New game detected! Solver will continue.")
            return True

        time.sleep(1)

    print("⚠️ Timeout waiting for new game. Skipping to next.")
    return False

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
