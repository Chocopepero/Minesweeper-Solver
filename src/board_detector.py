import cv2
import numpy as np
from typing import List, Tuple, Optional


class BoardDetector:
    """Detects and extracts the Minesweeper board from a screenshot."""
    DEFAULT_TOP_OFFSET = 101
    DEFAULT_SIDE_OFFSET = 15
    DEFAULT_BOTTOM_OFFSET = 43
    
    def __init__(self, cell_size: int = 16):
        self.cell_size = cell_size
        
    def detect_board_region(self, image: np.ndarray, 
                           top_offset: int = None, 
                           side_offset: int = None, 
                           bottom_offset: int = None) -> Optional[Tuple[int, int, int, int]]:
        if top_offset is None:
            top_offset = self.DEFAULT_TOP_OFFSET
        if side_offset is None:
            side_offset = self.DEFAULT_SIDE_OFFSET
        if bottom_offset is None:
            bottom_offset = self.DEFAULT_BOTTOM_OFFSET
            
        h, w = image.shape[:2]
        board_x = side_offset
        board_y = top_offset
        board_w = w - (2 * side_offset)
        board_h = h - top_offset - bottom_offset
        
        if board_w <= 0 or board_h <= 0:
            print(f"Invalid board region: {board_w}x{board_h}")
            return None
            
        return (board_x, board_y, board_w, board_h)
    
    def extract_board(self, image: np.ndarray, 
                     board_region: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = board_region
        return image[y:y+h, x:x+w]
    
    def auto_detect_board_size(self, board_image: np.ndarray) -> Tuple[int, int]:
        h, w = board_image.shape[:2]

        cols = w // self.cell_size
        rows = h // self.cell_size

        print(f"Auto-detected board size: {rows} rows × {cols} columns")
        return (rows, cols)

    def extract_cells(self, board_image: np.ndarray,
                     rows: int,
                     cols: int) -> List[List[np.ndarray]]:
        h, w = board_image.shape[:2]
        
        cell_h = self.cell_size
        cell_w = self.cell_size
        
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
                cell_y = row * cell_h
                cell_x = col * cell_w
                
                cell_y_end = min(cell_y + cell_h, h)
                cell_x_end = min(cell_x + cell_w, w)
                
                cell = board_image[cell_y:cell_y_end, cell_x:cell_x_end]
                row_cells.append(cell)
            
            cells.append(row_cells)
        
        return cells
    
    def visualize_grid(self, board_image: np.ndarray, 
                      rows: int, 
                      cols: int, 
                      output_path: str = "grid_overlay.png"):
        vis = board_image.copy()
        h, w = board_image.shape[:2]
        
        cell_h = self.cell_size
        cell_w = self.cell_size
        
        # Draw horizontal lines
        for row in range(rows + 1):
            y = row * cell_h
            if y <= h: 
                cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
        
        # Draw vertical lines
        for col in range(cols + 1):
            x = col * cell_w
            if x <= w:
                cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
        
        cv2.imwrite(output_path, vis)
        print(f"Grid visualization saved to {output_path}")
