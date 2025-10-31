"""
Board Detector - Converts screenshot to Minesweeper game board
Extracts individual cells from the captured image for processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class BoardDetector:
    """Detects and extracts the Minesweeper board from a screenshot."""
    
    # Calibrated offsets for Minesweeper Arbiter
    DEFAULT_TOP_OFFSET = 101
    DEFAULT_SIDE_OFFSET = 15
    DEFAULT_BOTTOM_OFFSET = 43
    
    def __init__(self, cell_size: int = 16):
        """
        Initialize the board detector.
        
        Args:
            cell_size: Size of each cell in pixels (default 16x16 for Arbiter)
        """
        self.cell_size = cell_size
        
    def detect_board_region(self, image: np.ndarray, 
                           top_offset: int = None, 
                           side_offset: int = None, 
                           bottom_offset: int = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the game board region in the screenshot.
        
        Args:
            image: Screenshot as numpy array (BGR format)
            top_offset: Pixels to skip from top (default: 101)
            side_offset: Pixels to skip from sides (default: 15)
            bottom_offset: Pixels to skip from bottom (default: 43)

        Returns:
            Tuple of (x, y, width, height) or None if not found
        """
        # Use calibrated defaults if not provided
        if top_offset is None:
            top_offset = self.DEFAULT_TOP_OFFSET
        if side_offset is None:
            side_offset = self.DEFAULT_SIDE_OFFSET
        if bottom_offset is None:
            bottom_offset = self.DEFAULT_BOTTOM_OFFSET
            
        h, w = image.shape[:2]
        
        # Apply offsets to get game board region
        board_x = side_offset
        board_y = top_offset
        board_w = w - (2 * side_offset)
        board_h = h - top_offset - bottom_offset
        
        # Validate region
        if board_w <= 0 or board_h <= 0:
            print(f"Invalid board region: {board_w}x{board_h}")
            return None
            
        return (board_x, board_y, board_w, board_h)
    
    def extract_board(self, image: np.ndarray, 
                     board_region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract just the board region from the full screenshot.
        
        Args:
            image: Full screenshot
            board_region: (x, y, width, height) of board
            
        Returns:
            Cropped board image
        """
        x, y, w, h = board_region
        return image[y:y+h, x:x+w]
    
    def extract_cells(self, board_image: np.ndarray, 
                     rows: int, 
                     cols: int) -> List[List[np.ndarray]]:
        """
        Extract individual cell images from the board.
        
        Args:
            board_image: Board region as numpy array
            rows: Number of rows in the board
            cols: Number of columns in the board
            
        Returns:
            2D list of cell images: cells[row][col] = cell_image
        """
        h, w = board_image.shape[:2]
        
        # Use the known cell size for more accurate extraction
        cell_h = self.cell_size
        cell_w = self.cell_size
        
        # Calculate expected board size
        expected_w = cols * cell_w
        expected_h = rows * cell_h
        
        print(f"Board size: {w}x{h} pixels")
        print(f"Expected size: {expected_w}x{expected_h} pixels")
        print(f"Cell size: {cell_w}x{cell_h} pixels (fixed)")
        print(f"Extracting {rows}x{cols} = {rows*cols} cells")
        
        if abs(w - expected_w) > 5 or abs(h - expected_h) > 5:
            print(f"⚠️  Warning: Board size mismatch! Expected {expected_w}x{expected_h}, got {w}x{h}")
            print(f"   Consider adjusting offsets or cell_size")
        
        cells = []
        for row in range(rows):
            row_cells = []
            for col in range(cols):
                # Calculate cell position using fixed cell size
                cell_y = row * cell_h
                cell_x = col * cell_w
                
                # Ensure we don't go out of bounds
                cell_y_end = min(cell_y + cell_h, h)
                cell_x_end = min(cell_x + cell_w, w)
                
                # Extract cell image
                cell = board_image[cell_y:cell_y_end, cell_x:cell_x_end]
                row_cells.append(cell)
            
            cells.append(row_cells)
        
        return cells
    
    def visualize_grid(self, board_image: np.ndarray, 
                      rows: int, 
                      cols: int, 
                      output_path: str = "grid_overlay.png"):
        """
        Draw grid lines over the board for debugging.
        
        Args:
            board_image: Board image
            rows: Number of rows
            cols: Number of columns
            output_path: Where to save the visualization
        """
        # Create a copy to draw on
        vis = board_image.copy()
        h, w = board_image.shape[:2]
        
        # Use fixed cell size for accurate grid lines
        cell_h = self.cell_size
        cell_w = self.cell_size
        
        # Draw horizontal lines
        for row in range(rows + 1):
            y = row * cell_h
            if y <= h:  # Only draw if within bounds
                cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
        
        # Draw vertical lines
        for col in range(cols + 1):
            x = col * cell_w
            if x <= w:  # Only draw if within bounds
                cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
        
        cv2.imwrite(output_path, vis)
        print(f"Grid visualization saved to {output_path}")


# Example usage
if __name__ == "__main__":
    from screengrabber import capture_minesweeper_board
    
    print("Capturing Minesweeper board...")
    board_image = capture_minesweeper_board()
    
    if board_image is None:
        print("Failed to capture board")
        exit(1)
    
    # Create detector
    detector = BoardDetector()
    
    # Detect board region using calibrated defaults (101, 15, 43)
    board_region = detector.detect_board_region(board_image)
    
    if board_region is None:
        print("Failed to detect board region")
        exit(1)
    
    print(f"Board region: {board_region}")
    
    # Extract just the board
    board = detector.extract_board(board_image, board_region)
    cv2.imwrite("extracted_board.png", board)
    # Extract individual cells (example: 8x8 beginner board)
    rows, cols = 8, 8
    cells = detector.extract_cells(board, rows, cols)
    
    # Visualize the grid
    detector.visualize_grid(board, rows, cols)
    
    # Save a few sample cells
    print("\nSaving sample cells...")
    cv2.imwrite("cell_0_0.png", cells[0][0])  # Top-left corner
    cv2.imwrite("cell_4_4.png", cells[4][4])  # Center
    cv2.imwrite("cell_7_7.png", cells[7][7])  # Bottom-right corner
    
    print(f"\n✅ Extracted {rows * cols} cells successfully!")
    print("Check the generated images to verify cell extraction")
