from __future__ import annotations

import os
import threading
import time

import cv2
import pyautogui
from tensorflow.keras.models import load_model

from src.config import mouse
from src.config import recheck
from src.config import stay_on_top
from src.handTrackerMod import handDetector

# Load the gesture recognizer model
path = os.path.join(os.getcwd() + "/model")
model = load_model(path)

# Load class names
class_name_path = os.path.join(os.path.dirname(path) + "/gesture.names")
f = open(class_name_path)
classNames = f.read().split("\n")
f.close()
print(classNames)

# FPS
fps_start_time = time.time()
fps = 0
frame_count = 0

current_gesture = None


def show_frames(className, classNames, frame, target, args):
    """
    This function displays the frames of a video stream with additional information.
    Parameters:
        className (str): The class name of the current gesture.
        classNames (list[str]): The list of all available gesture class names.
        frame (numpy array): The current frame of the video stream.
        target (function): The target function to be executed asynchronously.
        args (tuple): The arguments to be passed to the target function.
    Returns:
        None
    """

    global mouse
    global current_gesture

    if className == "stop":
        mouse = True

    elif className == "rock":
        mouse = False

    if className in classNames:
        current_gesture = className

    # show the prediction on the frame
    cv2.putText(
        frame,
        current_gesture,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    if recheck:
        # run threading so that check_class can be run asynchronously in the background.
        t = threading.Thread(target=target, args=args)
        t.start()  # start child thread

    # Display the mode on the frame
    if mouse:
        mode = "Cursor Mode"
    else:
        mode = "Gesture Mode"

    cv2.putText(
        frame,
        mode,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Show the final output
    x, y = pyautogui.size()
    frame = cv2.resize(frame, (int(x / 6), int(y / 4.5)))
    cv2.imshow("Hand Tracking", frame)

    # Set the display to always stay on top
    if stay_on_top:
        cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_TOPMOST, 1)

    return


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    detector = handDetector(cap)

    while True:
        ret, frame = cap.read()

        frame_count += 1
        if frame_count >= 10:
            fps_end_time = time.time()
            fps = int(frame_count / (fps_end_time - fps_start_time))
            frame_count = 0
            fps_start_time = fps_end_time

        if not ret:
            # Handle the case when no frame is captured (e.g., camera not available)
            print("Failed to capture a frame.")
        else:
            x, y, c = frame.shape
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            className = detector.findHands(frame, model, classNames, x, y)

            if mouse == True:
                args = (x, y, frame)
                frame = detector.mouse_control(*args)
                show_frames(
                    className,
                    classNames,
                    frame,
                    detector.mouse_control,
                    args,
                )
            else:
                args = (className,)
                show_frames(
                    className,
                    classNames,
                    frame,
                    detector.check_gesture,
                    args,
                )

            if cv2.waitKey(1) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                break
