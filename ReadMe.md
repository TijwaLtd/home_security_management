# YOLOv5 Webcam Detection with Object Capture

This project uses YOLOv5 for real-time object detection via a webcam. The script captures images of specified classes (`person`, `car`, etc.) when detected, along with their dominant colors. Captured images are saved locally.

## Features

- Real-time object detection using YOLOv5.
- Capture images of specified object classes.
- Analyze and label objects with their dominant colors.
- Save detected frames to the `captures` folder.
- Manual saving of frames by pressing `s` during execution.

---

## Prerequisites

### Python Libraries

Ensure you have the following Python libraries installed:

- **PyTorch**: Required for YOLOv5.
- **OpenCV**: For video frame processing.
- **NumPy**: For numerical operations.
- **Pillow**: For image handling.
- **SciPy**: For color distance calculation.

Install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```



This error happens because OpenCV was installed without GUI support.

## Steps to Fix:

1. **Install required dependencies** (for Ubuntu/Debian):

   Run this command to install the necessary libraries:

   ```bash
   sudo apt-get install libgtk2.0-dev libgtk-3-dev pkg-config
