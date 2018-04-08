# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:14:28 2018

@author: Nishant Ghanate
"""

import cv2 
from datetime import datetime


threshold = 81500
camera = cv2.VideoCapture(0)

WindowName = "Movement Indicator"



timeCheck = datetime.now().strftime('%Ss')

while True:
    ret,frame = camera.read()
    cv2.imShow(WindowName,frame)

    
    key = cv2.waitkey(10)
    if key == 27:
        cv2.destroyWindow(WindowName)
        break