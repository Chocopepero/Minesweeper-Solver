# Minesweeper-Solver

CPSC481 AI Project that uses Computer Vision to detect and solve a minesweeper board.

This project automatically plays Minesweeper Arbiter by:
1. Capturing the game window using computer vision
2. Classifying each cell's state (unrevealed, numbers, flags, mines)
3. Applying logical deduction rules to find safe cells and mines
4. Automatically clicking cells and flagging mines

## Features

- ✅ Real-time board detection and classification
- ✅ Advanced pattern recognition algorithms
- ✅ Automatic mouse control for gameplay
- ✅ Multiple solving strategies with intelligent fallback
- ✅ Works with any board size (auto-detection)

## Requirements

- Python 3.10+
- Minesweeper Arbiter (or compatible Minesweeper game)
- Windows OS (for win32 API support)

## Setup Instructions

### 1. Install Python

If you don't have Python installed:

1. Download Python 3.10 or higher from [python.org](https://www.python.org/downloads/)
2. Verify installation by opening a terminal/command prompt and running:
   ```bash
   python --version
   ```

### 2. Download This Project

Clone or download this repository:

```bash
git clone https://github.com/Chocopepero/Minesweeper-Solver.git
cd Minesweeper-Solver
```

Or download as ZIP and extract it to a folder of your choice.

### 3. Install Python Dependencies

Open a terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

This will install:
- `mss` - Screen capture library
- `opencv-python` - Image processing
- `numpy` - Numerical computations
- `pygetwindow` - Window management
- `pillow` - Image handling
- `pywin32` - Windows API access

### 4. Install Minesweeper Arbiter

1. Download Minesweeper Arbiter from [minesweeper.info](http://www.minesweeper.info/downloads/WinmineArbiter.html)
2. Install and launch the application
3. The solver is calibrated for Arbiter's default window size and cell dimensions

You're all set! Follow the usage instructions below.

## Usage

### Basic Usage

1. **Open Minesweeper Arbiter** and make sure the window is visible on screen with no overlapping apps

2. **Run the analyzer** to view the current board state:
```bash
python main.py
```

This will:
- Capture the game window
- Auto-detect board dimensions
- Classify all cells
- Display the board state
- Ask if you want to auto-solve

3. **Auto-solve** by typing `y` when prompted, or just analyze by typing `n`

### Advanced Usage

To run only the solver (without initial analysis):
```bash
python solve.py
```