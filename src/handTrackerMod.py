from __future__ import annotations

from time import sleep

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import win32api

from src.config import distance_x
from src.config import distance_y
from src.config import keystrokes
from src.config import recheck

pyautogui.FAILSAFE = False


class handDetector:
    """
    Initializes the class with the given parameters.
    Parameters:
        cap (object): The video capture object.
    Returns:
        None
    """

    def __init__(self, cap):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.cap = cap
        self.results = None
        self.mouse_down_state = False
        self.frame_count = 0
        self.distance_x = distance_x
        self.distance_y = distance_y

    def findHands(self, frame, model, classNames, x, y):
        """
        Find hands in the given frame using the specified model and classNames.
        Parameters:
            frame (numpy.ndarray): The frame in which hands are to be detected.
            model (Model): The hand detection model to be used.
            classNames (List[str]): The list of class names for hand detection.
        Returns:
            str: The name of the detected hand class.
        """
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand landmark prediction
        self.results = self.hands.process(framergb)

        className = ""

        if self.results.multi_hand_landmarks:
            landmarks = []
            for handLms in self.results.multi_hand_landmarks:
                for lm in handLms.landmark:
                    lmx = int((1 - lm.x) * y)  # Flip the landmarks horizontally
                    lmy = int(lm.y * x)
                    landmarks.append([lmx, lmy])

                self.mpDraw.draw_landmarks(
                    frame,
                    handLms,
                    self.mpHands.HAND_CONNECTIONS,
                    self.mpDraw.DrawingSpec(
                        color=(250, 44, 250),
                        thickness=2,
                        circle_radius=2,
                    ),
                )

                self.frame_count += 1
                print(self.frame_count)
                if self.frame_count >= 5 and not self.mouse_down_state:
                    self.frame_count = 0

                    # Predict gesture in Hand Gesture Recognition project
                    prediction = model.predict([landmarks])
                    print(prediction)
                    classID = np.argmax(prediction)
                    className = classNames[classID]

        return className

    def check_gesture(self, className):
        """
        Check the given gesture and perform the corresponding action.
        Parameters:
        - className (str): The name of the gesture to be checked.
        Returns:
        - None
        """
        global recheck

        if className == "thumbs down" and recheck:
            pyautogui.press(keystrokes["thumbs_down"])
            recheck = False
            sleep(keystrokes["sleep"])
            recheck = True
        elif (className == "thumbs up" or className == "call me") and recheck:
            pyautogui.press(keystrokes["thumbs_up"])
            recheck = False
            sleep(keystrokes["sleep"])
            recheck = True
        elif (
            className in ("okay", "stop", "live long", "fist", "smile", "peace", "rock")
            and recheck
        ):
            recheck = False
            sleep(keystrokes["sleep"])
            recheck = True

    def mouse_control(self, x, y, frame):
        """
        Controls the mouse based on the hand landmarks detected in the frame.
        Args:
            x (int): The width of the frame.
            y (int): The height of the frame.
            frame (numpy.ndarray): The input frame.
        Returns:
            numpy.ndarray: The modified frame with the mouse control applied.
        """
        # Define the top right quadrant dimensions (50% of x and 50% of y)
        quadrant_width = x // 2
        quadrant_height = y // 2

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                mouse_point = handLandmarks.landmark[
                    self.mpHands.HandLandmark.MIDDLE_FINGER_MCP
                ]
                indexfingertip = handLandmarks.landmark[
                    self.mpHands.HandLandmark.INDEX_FINGER_TIP
                ]
                thumbfingertip = handLandmarks.landmark[
                    self.mpHands.HandLandmark.THUMB_TIP
                ]

                # Continue only if the mouse_point is in the top right quadrant
                if (
                    mouse_point.x * x > quadrant_width
                    and mouse_point.y * y < quadrant_height
                ):
                    indexfingertip_x, indexfingertip_y = int(
                        indexfingertip.x * x,
                    ), int(indexfingertip.y * y)
                    thumbfingertip_x, thumbfingertip_y = int(
                        thumbfingertip.x * x,
                    ), int(thumbfingertip.y * y)

                    # Transformed coordinate system for mapping the cursor
                    cursor_x = int(
                        (mouse_point.x - 0.5) * 2 * win32api.GetSystemMetrics(0),
                    )
                    cursor_y = int(mouse_point.y * 2 * win32api.GetSystemMetrics(1))

                    cursor_x = max(0, min(cursor_x, win32api.GetSystemMetrics(0) - 1))
                    cursor_y = max(0, min(cursor_y, win32api.GetSystemMetrics(1) - 1))

                    win32api.SetCursorPos((cursor_x, cursor_y))

                    Distance_x = abs(indexfingertip_x - thumbfingertip_x)
                    Distance_y = abs(indexfingertip_y - thumbfingertip_y)

                    # Check for "CLICK DOWN" condition first
                    if Distance_x < self.distance_x and Distance_y < self.distance_y:
                        if not self.mouse_down_state:
                            print("CLICK DOWN")
                            pyautogui.mouseDown(button="left")
                            self.mouse_down_state = True

                        if self.mouse_down_state:
                            cv2.putText(
                                frame,
                                "CLICK DOWN",
                                (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                    # If none of the conditions are met and the mouse is in down state, release it.
                    elif self.mouse_down_state:
                        print("MOUSE UP")
                        pyautogui.mouseUp(button="left")
                        self.mouse_down_state = False

        return frame
