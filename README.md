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

### Basic Usage

1. **Open Minesweeper Arbiter** and make sure the window is visible on screen

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

## How It Works

### 1. Computer Vision Pipeline

**Screen Capture:**
- Uses Windows API (`win32gui`) to capture the Minesweeper Arbiter window
- Works regardless of window position on screen

**Board Detection:**
- Auto-detects board dimensions based on window size
- Uses calibrated offsets for Minesweeper Arbiter (101px top, 15px sides, 43px bottom)
- Extracts individual 16×16 pixel cells

**Cell Classification:**
- Analyzes color patterns and brightness to classify each cell
- Detects: unrevealed cells, numbers (1-8), flags, mines, and empty cells
- Uses color matching for number detection (each number has distinct color)

### 2. Solving Algorithm

The solver applies rules in this order:

#### **Basic Patterns**

1. **Rule 1 - Safe Cells:**
   - If a number cell has exactly that many flags around it, all other unrevealed neighbors are safe

2. **Rule 2 - Obvious Mines:**
   - If a number cell has exactly that many unrevealed cells around it, they're all mines

#### **Advanced Patterns**

3. **Corner 1 Pattern:**
   ```
   [1] U    ← Corner position
    U  U
   ```
   - A corner 1 has only 3 neighbors
   - If only one neighbor is unrevealed, it must be a mine

4. **Subset Overlap Pattern:**
   ```
   [1][2]
    U UU
   ```
   - If N(A) ⊆ N(B), then mines in N(B) \ N(A) = number(B) - number(A)
   - If the difference equals the count, those cells are all mines
   - If the difference is 0, those cells are all safe
   - **This is the most powerful pattern** - works with any number combination!

5. **1-2-1 Pattern:**
   ```
   [1][2][1]
    U  U  U
   ```
   - The middle unrevealed cell (above the 2) is a mine
   - The outer cells (above the 1s) are safe

6. **1-2-2-1 Pattern:**
   ```
   [1][2][2][1]
    U  U  U  U
   ```
   - The two middle unrevealed cells (above the 2s) are mines
   - The outer cells (above the 1s) are safe

#### **Guessing Strategy**

If no patterns match:
- Picks the unrevealed cell with the most revealed neighbors (statistically safer)
- Prefers cells near known safe areas

### 3. Mouse Control

- Calculates exact pixel coordinates for each cell
- Left-click to reveal cells
- Right-click to place flags
- Optimized delays for fast solving (~20ms between clicks)


### Success Rate
Success rate depends on:
- Board layout (some boards are logically unsolvable without guessing)
- Classification accuracy
- Pattern complexity