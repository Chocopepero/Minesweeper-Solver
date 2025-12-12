import numpy as np
import cv2
import win32gui
import win32ui
import win32con
from PIL import Image

def find_minesweeper_window():
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "Arbiter" in title:
                windows.append(hwnd)
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)

    if not windows:
        print("Could not find the Minesweeper Arbiter window.")
        return None

    hwnd = windows[0]
    title = win32gui.GetWindowText(hwnd)
    rect = win32gui.GetWindowRect(hwnd)

    print(f"Found window: {title}")
    print(f"Position: x={rect[0]}, y={rect[1]}")
    print(f"Size: width={rect[2]-rect[0]}, height={rect[3]-rect[1]}")

    return hwnd
    
def capture_minesweeper_board():
    """
    Capture a screenshot of the Minesweeper Arbiter window using win32gui.
    """
    hwnd = find_minesweeper_window()
    if hwnd is None:
        print("Cannot capture screenshot: Minesweeper window not found.")
        return None

    try:
        # Get window dimensions
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        # Get the window's device context
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        # Create a bitmap
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        # Copy the window content to the bitmap
        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

        # Convert to PIL Image
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        pil_image = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1
        )

        # Convert to numpy array (OpenCV format)
        board_image = np.array(pil_image)
        board_image = cv2.cvtColor(board_image, cv2.COLOR_RGB2BGR)

        # Clean up
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        return board_image

    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    capture_minesweeper_board()