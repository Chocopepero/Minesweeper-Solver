"""
Mouse Controller - Clicks on cells in the Minesweeper game
"""

import win32gui
import win32api
import win32con
import time
from typing import Tuple, List


class MouseController:
    """Controls mouse clicks on the Minesweeper window."""

    def __init__(self, hwnd, board_x: int, board_y: int, cell_size: int = 16):
        """
        Initialize the mouse controller.

        Args:
            hwnd: Window handle of Minesweeper
            board_x: X offset of board from window left
            board_y: Y offset of board from window top
            cell_size: Size of each cell in pixels
        """
        self.hwnd = hwnd
        self.board_x = board_x
        self.board_y = board_y
        self.cell_size = cell_size

    def get_cell_center(self, row: int, col: int) -> Tuple[int, int]:
        """
        Get the screen coordinates of the center of a cell.

        Args:
            row: Row index
            col: Column index

        Returns:
            (x, y) screen coordinates
        """
        # Calculate position relative to board
        cell_x = col * self.cell_size + self.cell_size // 2
        cell_y = row * self.cell_size + self.cell_size // 2

        # Add board offset
        board_rel_x = self.board_x + cell_x
        board_rel_y = self.board_y + cell_y

        # Convert to screen coordinates
        window_rect = win32gui.GetWindowRect(self.hwnd)
        screen_x = window_rect[0] + board_rel_x
        screen_y = window_rect[1] + board_rel_y

        return (screen_x, screen_y)

    def click_cell(self, row: int, col: int, right_click: bool = False):
        """
        Click on a cell in the game.

        Args:
            row: Row index
            col: Column index
            right_click: If True, right-click (to flag). If False, left-click (to reveal)
        """
        x, y = self.get_cell_center(row, col)

        # Save current cursor position
        original_pos = win32api.GetCursorPos()

        # Move cursor to the cell
        win32api.SetCursorPos((x, y))
        time.sleep(0.01)  # Reduced delay

        # Perform the click
        if right_click:
            # Right click (flag)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0)
            time.sleep(0.005)
            win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)
        else:
            # Left click (reveal)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
            time.sleep(0.005)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

        time.sleep(0.01)  # Reduced delay after click

        # Restore cursor position
        win32api.SetCursorPos(original_pos)

    def click_cells(self, cells: list, right_click: bool = False, delay: float = 0.1, verbose: bool = False):
        """
        Click multiple cells with a delay between each.

        Args:
            cells: List of (row, col) tuples
            right_click: If True, right-click all cells. If False, left-click
            delay: Delay in seconds between clicks
            verbose: If True, print each click
        """
        for row, col in cells:
            if verbose:
                print(f"   {'Flagging' if right_click else 'Clicking'} cell ({row}, {col})")
            self.click_cell(row, col, right_click)
            time.sleep(delay)


if __name__ == "__main__":
    print("Mouse Controller Module")
    print("Import this module to control mouse clicks on Minesweeper.")
