from main import run_solver, wait_for_new_game
from src.screengrabber import capture_minesweeper_board
from src.board_detector import BoardDetector
from src.cell_classifier import CellClassifier
import time


# Detect board once before loop
detector = BoardDetector()
classifier = CellClassifier()
board_image = capture_minesweeper_board()
board_region = detector.detect_board_region(board_image)

# Run AI 20 times for each difficulty
configs = [
    ("beginner", 9, 9),
    ("intermediate", 16, 16),
    ("expert", 16, 30)
]

for difficulty, rows, cols in configs:
    print(f"\n===============================")
    print(f"🎯 Starting {difficulty.title()} difficulty")
    print("===============================")
    
    for game_number in range(1, 21):
        print(f"\n🏁 Starting Game {game_number} - {difficulty.title()}")
        time.sleep(2)
        run_solver(rows, cols, board_region)

        wait_for_new_game(detector, classifier, rows, cols)
