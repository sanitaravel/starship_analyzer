import cv2
import numpy as np

def display_image(image: np.ndarray, text: str) -> None:
    """
    Display an image in a window.

    Args:
        image (numpy.ndarray): The image to display.
        text (str): The text to display in the window title.
    """
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()