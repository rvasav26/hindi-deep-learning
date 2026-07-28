from __future__ import annotations

import time

import cv2
import mediapipe as mp

FILLED = cv2.FILLED
LANDMARK_COLOR = (245, 135, 66)
LANDMARK_RADIUS = 10


class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 2,
        modelComplexity: int = 1,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
    ):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplexity,
            self.detectionCon,
            self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw: bool = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo: int = 0, draw: bool = True) -> list[list[int]]:
        lmList: list[list[int]] = []

        if self.results and self.results.multi_hand_landmarks:
            if handNo >= len(self.results.multi_hand_landmarks):
                return lmList

            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for landmark_id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([landmark_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), LANDMARK_RADIUS, LANDMARK_COLOR, FILLED)

        return lmList
