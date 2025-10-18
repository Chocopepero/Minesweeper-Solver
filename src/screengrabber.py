import pygetwindow as gw
import pyautogui

def find_minesweeper_window() -> tuple[int, int, int, int]:
    """
    Locate the Minesweeper game window on the screen.

    Returns:
        A tuple (x, y, width, height) representing the window's position and size,
        or None if the window is not found.
    """

    try:
        windows = gw.getWindowsWithTitle("Arbiter")
        
        if not windows:
            print("Could not find the Minesweeper Arbiter window.")
            return None
        
        # The first matching window
        window = windows[0]
        window.activate()  # Bring window to front (may fail silently)

        print(f"Found window: {window.title}")
        print(f"Position: x={window.left}, y={window.top}")
        print(f"Size: width={window.width}, height={window.height}")

        return (window.left, window.top, window.width, window.height)

    except Exception as e:
        print(f"Error finding Minesweeper window: {e}")
        return None
    
def capture_minesweeper_board() -> None:
    """
    Capture a screenshot of the Minesweeper Arbiter window for debug purposes.
    """
    window_info = find_minesweeper_window()
    if window_info is None:
        print("Cannot capture screenshot: Minesweeper window not found.")
        return

    x, y, width, height = window_info
   
    board_image = pyautogui.screenshot(region=(x, y, width, height))
    board_image.save("minesweeper_board_debug.png")
    print("Screenshot saved as minesweeper_board_debug.png")

if __name__ == "__main__":
    capture_minesweeper_board()