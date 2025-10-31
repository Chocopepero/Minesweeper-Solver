"""
Cell classification for Minesweeper.
Identifies the state of each cell (unrevealed, revealed, number, flag, mine, etc.)
"""

import cv2
import numpy as np
from enum import Enum
from typing import Dict, Tuple


class CellState(Enum):
    """Possible states of a Minesweeper cell."""
    UNREVEALED = "unrevealed"
    REVEALED_0 = "0"
    REVEALED_1 = "1"
    REVEALED_2 = "2"
    REVEALED_3 = "3"
    REVEALED_4 = "4"
    REVEALED_5 = "5"
    REVEALED_6 = "6"
    REVEALED_7 = "7"
    REVEALED_8 = "8"
    FLAG = "flag"
    MINE = "mine"
    UNKNOWN = "unknown"


class CellClassifier:
    """Classifies Minesweeper cells using color analysis."""
    
    def __init__(self):
        """Initialize the classifier with color templates."""
        # BGR color ranges for different cell types
        self.color_ranges = {
            # Unrevealed cells (light gray, raised)
            CellState.UNREVEALED: {
                'lower': np.array([180, 180, 180]),
                'upper': np.array([200, 200, 200])
            },
            # Revealed empty cells (darker gray, flat)
            CellState.REVEALED_0: {
                'lower': np.array([180, 180, 180]),
                'upper': np.array([195, 195, 195])
            },
            # Flag (red)
            CellState.FLAG: {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([100, 100, 255])
            },
            # Mine (black center)
            CellState.MINE: {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([50, 50, 50])
            }
        }
        
        # Number colors (BGR format) - calibrated from actual Arbiter colors
        # Numbers 1-3 are calibrated, 4-8 are Minesweeper standard colors
        self.number_colors = {
            CellState.REVEALED_1: (199, 55, 55),      # Blue
            CellState.REVEALED_2: (41, 128, 41),      # Green
            CellState.REVEALED_3: (42, 42, 212),      # Red
            CellState.REVEALED_4: (128, 0, 0),        # Dark blue/navy
            CellState.REVEALED_5: (0, 0, 128),        # Maroon/dark red
            CellState.REVEALED_6: (128, 128, 0),      # Cyan/teal
            CellState.REVEALED_7: (0, 0, 0),          # Black
            CellState.REVEALED_8: (128, 128, 128),    # Gray
        }
    
    def classify_cell(self, cell_image: np.ndarray, debug: bool = False) -> CellState:
        """
        Classify a single cell image.
        
        Args:
            cell_image: Cell image as numpy array (BGR)
            debug: Print debug information
            
        Returns:
            CellState enum value
        """
        if cell_image is None or cell_image.size == 0:
            return CellState.UNKNOWN
        
        # Get average color and brightness
        avg_color = cv2.mean(cell_image)[:3]  # BGR
        avg_brightness = np.mean(cell_image)
        
        if debug:
            print(f"  Avg BGR: {[int(c) for c in avg_color]}, Brightness: {avg_brightness:.1f}")
        
        # Check for flags FIRST - they have red color that distinguishes them from everything else
        if self._is_flag(cell_image):
            if debug: print("  -> FLAG")
            return CellState.FLAG
        
        # Check if it's unrevealed (light gray with 3D effect)
        if self._is_unrevealed(cell_image):
            if debug: print("  -> UNREVEALED")
            return CellState.UNREVEALED
        
        # Check if it's revealed but empty (must check before numbers)
        if self._is_revealed_empty(cell_image):
            if debug: print("  -> REVEALED_0")
            return CellState.REVEALED_0
        
        # Check for numbers (note: number 3 is red but has different characteristics than flags)
        number_state = self._detect_number(cell_image)
        if number_state != CellState.UNKNOWN:
            if debug: print(f"  -> {number_state.value}")
            return number_state
        
        # Check if it's a mine (has black center)
        if self._has_color_match(cell_image, CellState.MINE):
            if debug: print("  -> MINE")
            return CellState.MINE
        
        if debug: print("  -> UNKNOWN")
        return CellState.UNKNOWN
    
    def _is_unrevealed(self, cell_image: np.ndarray) -> bool:
        """Check if cell is unrevealed by looking for raised 3D effect."""
        # Unrevealed cells have contrast between corners due to 3D shading
        # In Arbiter, bottom-right is brighter (highlight) than top-left
        h, w = cell_image.shape[:2]
        
        if h < 3 or w < 3:
            return False
        
        # Sample corners
        top_left = cell_image[0:max(1, h//4), 0:max(1, w//4)]
        bottom_right = cell_image[max(1, 3*h//4):h, max(1, 3*w//4):w]
        
        tl_brightness = np.mean(top_left)
        br_brightness = np.mean(bottom_right)
        
        # Unrevealed cells are bright overall and have contrast
        avg_brightness = np.mean(cell_image)
        std_dev = np.std(cell_image)
        
        # Check for STRONG 3D effect: unrevealed cells have much stronger contrast
        # Empty revealed cells: ~28 difference, ~21 std dev
        # Unrevealed cells: ~95 difference, ~41 std dev
        is_bright = avg_brightness > 160
        has_strong_contrast = abs(tl_brightness - br_brightness) > 50  # Much stronger 3D
        has_strong_variation = std_dev > 30  # Unrevealed cells have higher variation
        
        # CRITICAL: Unrevealed cells are mostly gray (no strong colors)
        # Check that the cell doesn't have significant color variation (no numbers)
        avg_color = cv2.mean(cell_image)[:3]  # BGR
        color_std = np.std([avg_color[0], avg_color[1], avg_color[2]])
        is_uniform_gray = color_std < 5  # Unrevealed cells are pure gray
        
        return is_bright and has_strong_contrast and has_strong_variation and is_uniform_gray
    
    def _is_revealed_empty(self, cell_image: np.ndarray) -> bool:
        """Check if cell is revealed but empty (no number)."""
        # Revealed empty cells are uniform medium-gray
        # From debug: Empty cells have brightness ~184, std dev ~21
        avg_brightness = np.mean(cell_image)
        std_dev = np.std(cell_image)
        
        # Revealed empty cells are flatter and more uniform than unrevealed
        # Empty cells: brightness 180-190, std dev 15-25
        is_medium_gray = 175 < avg_brightness < 195
        is_uniform = std_dev < 30  # Less variation than unrevealed cells (which have ~41)
        
        # Also check that there's no colored pixels (no number)
        gray_mask = cv2.inRange(cell_image, np.array([150, 150, 150]), np.array([210, 210, 210]))
        gray_ratio = np.count_nonzero(gray_mask) / gray_mask.size
        
        return is_medium_gray and is_uniform and gray_ratio > 0.8
    
    def _is_flag(self, cell_image: np.ndarray) -> bool:
        """
        Check if cell is a flag by looking for specific flag pattern.
        Flags have: red pixels and reddish overall tint.
        """
        h, w = cell_image.shape[:2]
        
        # Get average color - flags have reddish tint, unrevealed cells are pure gray
        avg_color = cv2.mean(cell_image)[:3]  # BGR
        avg_b, avg_g, avg_r = avg_color
        
        # Check if cell has reddish tint (red channel higher than blue/green)
        has_red_tint = (avg_r > avg_g + 5) and (avg_r > avg_b + 5)
        
        if not has_red_tint:
            return False
        
        # Check for red pixels (flag itself)
        # Red in BGR: high R (index 2), significantly higher than B and G
        b, g, r = cv2.split(cell_image)
        
        # Look for actual red pixels (not white highlights)
        red_mask = (r > 200) & (r > b + 40) & (r > g + 40)
        red_count = np.count_nonzero(red_mask)
        
        # Flags should have some red pixels
        has_red_pixels = red_count > 20
        
        if not has_red_pixels:
            return False
        
        # Additional check: flags are on 3D cells (unrevealed background)
        has_3d = self._has_3d_effect(cell_image)
        
        # Check average brightness - unrevealed cells are brighter
        avg_brightness = np.mean(cell_image)
        is_bright = avg_brightness > 160
        
        # Flag if: has red tint AND has red pixels AND (has 3D effect OR is bright)
        return has_red_tint and has_red_pixels and (has_3d or is_bright)
    
    def _has_3d_effect(self, cell_image: np.ndarray) -> bool:
        """Check if cell has 3D raised effect (for flags/unrevealed)."""
        h, w = cell_image.shape[:2]
        if h < 4 or w < 4:
            return False
        
        # Check edges for light/dark pattern
        top_edge = np.mean(cell_image[0:max(1, h//4), :])
        bottom_edge = np.mean(cell_image[max(1, 3*h//4):h, :])
        
        # 3D cells have bright top, darker bottom
        return top_edge > bottom_edge + 5
    
    def _has_color_match(self, cell_image: np.ndarray, state: CellState) -> bool:
        """Check if cell matches a color range."""
        if state not in self.color_ranges:
            return False
        
        color_range = self.color_ranges[state]
        mask = cv2.inRange(cell_image, color_range['lower'], color_range['upper'])
        
        # If significant portion matches, return True
        match_ratio = np.count_nonzero(mask) / mask.size
        return match_ratio > 0.1  # At least 10% match
    
    def _detect_number(self, cell_image: np.ndarray) -> CellState:
        """
        Detect which number is in the cell by color analysis.
        
        Args:
            cell_image: Cell image as numpy array
            
        Returns:
            CellState for the detected number, or UNKNOWN
        """
        h, w = cell_image.shape[:2]
        if h < 4 or w < 4:
            return CellState.UNKNOWN
        
        # Extract center region where number would be (slightly larger area)
        center = cell_image[h//5:4*h//5, w//5:4*w//5]
        
        if center.size == 0:
            return CellState.UNKNOWN
        
        # Check if this looks like a revealed cell (gray background)
        # Numbers can significantly darken the cell center, so use minimal threshold
        avg_brightness = np.mean(center)
        if avg_brightness < 80 or avg_brightness > 220:
            return CellState.UNKNOWN
        
        # Find non-gray pixels (numbers are colored) - broader range
        gray_mask = cv2.inRange(center, np.array([160, 160, 160]), np.array([210, 210, 210]))
        non_gray = cv2.bitwise_not(gray_mask)
        non_gray_count = np.count_nonzero(non_gray)
        
        # If there are colored pixels, try to identify the number
        if non_gray_count > 3:  # At least some colored pixels
            # Get the dominant non-gray color
            colored_pixels = center[non_gray > 0]
            if len(colored_pixels) > 0:
                avg_color = np.mean(colored_pixels, axis=0)
                
                # Find closest matching number color
                min_distance = float('inf')
                best_match = CellState.UNKNOWN
                
                for state, color in self.number_colors.items():
                    distance = np.linalg.norm(np.array(color) - avg_color)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = state
                
                # Lenient threshold for color similarity (increased to handle variations)
                if min_distance < 120:
                    return best_match
        
        return CellState.UNKNOWN
    
    def classify_board(self, cells: list) -> list:
        """
        Classify all cells in a board.
        
        Args:
            cells: 2D list of cell images [row][col]
            
        Returns:
            2D list of CellState values [row][col]
        """
        board_state = []
        for row in cells:
            state_row = []
            for cell in row:
                state = self.classify_cell(cell)
                state_row.append(state)
            board_state.append(state_row)
        return board_state
    
    def print_board_state(self, board_state: list) -> None:
        """Print a visual representation of the board state."""
        symbol_map = {
            CellState.UNREVEALED: '□',
            CellState.REVEALED_0: '·',
            CellState.REVEALED_1: '1',
            CellState.REVEALED_2: '2',
            CellState.REVEALED_3: '3',
            CellState.REVEALED_4: '4',
            CellState.REVEALED_5: '5',
            CellState.REVEALED_6: '6',
            CellState.REVEALED_7: '7',
            CellState.REVEALED_8: '8',
            CellState.FLAG: '⚑',
            CellState.MINE: '💣',
            CellState.UNKNOWN: '?',
        }
        
        print("\nBoard State:")
        print("─" * (len(board_state[0]) * 2 + 1))
        for row in board_state:
            print(" ".join(symbol_map.get(cell, '?') for cell in row))
        print("─" * (len(board_state[0]) * 2 + 1))


if __name__ == "__main__":
    print("Cell Classifier Module")
    print("Import this module to classify Minesweeper cells.")
