import os
import subprocess
import sys

def check_tesseract_installed() -> None:
    """
    Check if Tesseract OCR is installed.

    Raises:
        subprocess.CalledProcessError: If Tesseract is not installed.
    """
    try:
        subprocess.check_call(["tesseract", "-v"])
        print("\033[92mTesseract is installed.\033[0m")  # Green text
    except subprocess.CalledProcessError:
        print("\033[91mTesseract is not installed. Please install it from https://github.com/tesseract-ocr/tesseract\033[0m")  # Red text

def create_virtualenv() -> None:
    """
    Create a virtual environment.
    """
    try:
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("\033[92mVirtual environment has been created.\033[0m")  # Green text
    except subprocess.CalledProcessError:
        print("\033[91mVirtual environment wasn't created! Something went HORRIBLTY wrong.\033[0m")  # Red text

def install_dependencies() -> None:
    """
    Install the required dependencies.
    """
    try:
        subprocess.check_call([os.path.join("venv", "Scripts", "pip"), "install", "-r", "requirements.txt"])
        print("\033[92mPackages were installed.\033[0m")  # Green text
    except subprocess.CalledProcessError:
        print("\033[91mVPackages weren't installed!. Something went HORRIBLTY wrong.\033[0m")  # Red text

def main() -> None:
    """
    Main function to set up the project.
    """
    check_tesseract_installed()
    create_virtualenv()
    install_dependencies()
    print("\033[92mSetup complete. To activate the virtual environment, run:\033[0m")  # Green text
    print("\033[93mvenv\\Scripts\\activate\033[0m" if os.name == "nt" else "\033[93msource venv/bin/activate\033[0m")  # Yellow text

if __name__ == "__main__":
    main()