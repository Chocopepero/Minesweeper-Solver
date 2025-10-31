import pygetwindow as gw
import mss
import numpy as np
import cv2

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
    
def capture_minesweeper_board():
    """
    Capture a screenshot of the Minesweeper Arbiter window for debug purposes.
    """
    window_info = find_minesweeper_window()
    if window_info is None:
        print("Cannot capture screenshot: Minesweeper window not found.")
        return None

    x, y, width, height = window_info
    
    # Ensure coordinates are valid (MSS doesn't like negative coords or off-screen)
    x = max(0, x)
    y = max(0, y)
    
    print(f"Capturing region: x={x}, y={y}, width={width}, height={height}")

    with mss.mss() as sct:
        monitor = {
            "left": x,
            "top": y,
            "width": width,
            "height": height
        }
        
        try:
            screenshot = sct.grab(monitor)
            board_image = np.array(screenshot)
            board_image = cv2.cvtColor(board_image, cv2.COLOR_BGRA2BGR)

            # Comment out the 2 lines below if you don't want to save the screenshot
            cv2.imwrite("minesweeper_board_debug.png", board_image)
            print("Screenshot saved as minesweeper_board_debug.png")

            print(board_image)  # Debug: print shape of captured image
            return board_image
        
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            print("Tip: Make sure the window is fully on-screen")
            return None

if __name__ == "__main__":
    capture_minesweeper_board()