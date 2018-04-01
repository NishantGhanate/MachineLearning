# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:14:28 2018

@author: Nishant Ghanate
"""

import cv2 
from datetime import datetime

def ImageDifference(t0,t1,t2):
    d1 = cv2.absdiff(t2,t1)
    d2 = cv2.absdiff(t1,t0)
    
threshold = 81500
camera = cv2.VideoCapture(0)

WindowName = "Movement Indicator"

t_minus = cv2.cvtColor(camera.read()[1] , cv2.COLOR_RBG2GRAY)
t =  cv2.cvtColor(camera.read()[1] , cv2.COLOR_RBG2GRAY)
t_plus =  cv2.cvtColor(camera.read()[1] , cv2.COLOR_RBG2GRAY)

timeCheck = datetime.now().strftime('%Ss')

while True:
    cv2.imShow(WindowName,camera.read()[1])
    if cv2.countNonZero(ImageDifference(t_minus , t , t_plus)) > threshold and timeCheck !=  datetime.now().strftime('%Ss'): 
        dimg = camera.read()[1]
        cv2.imWrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg' , dimg)
    timeCheck = datetime.now().strftime('%Ss')
    t_minus = t
    t = t_plus
    t_plus =  cv2.cvtColor(camera.read()[1] , cv2.COLOR_RBG2GRAY)
    
    key = cv2.waitkey(10)
    if key == 27:
        cv2.destroyWindow(WindowName)
        break