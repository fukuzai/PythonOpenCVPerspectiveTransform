# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:25:18 2019

@author: PCmgr_2E00109296

台形補正
https://qiita.com/mix_dvd/items/5674f26af467098842f0
"""
import cv2
import numpy as np

# 画像読み込み
img = cv2.imread('Image.jpg')
cv2.imshow('org', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# モノクロ変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
cv2.imwrite('Gray.jpg', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#二値化
ret, Bin= cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
cv2.imshow('Bin', Bin)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 輪郭抽出
image, contours, hierarchy = cv2.findContours(Bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 面積の大きいもののみ選別
areas = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10000:
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        areas.append(approx)

cv2.drawContours(img,areas,-1,(0,255,0),3)
cv2.imshow('Contours', img)
cv2.imwrite('Contours.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 射影変換
# 枠の各点を対応する座標にあわせて射影変換します
dst = []

pts1 = np.float32(areas[0])
pts2 = np.float32([[600,300],[600,0],[0,0],[0,300]])
# この座標の場合、600x300pixの画像に変換される

#----------
# getPerspectiveTransform : 4組の対応点から透視変換を表す 3×3 の行列を求める
# src – 入力画像上の四角形の頂点の座標
# dst – 出力画像上の対応する四角形の頂点の座標

# warpPerspective : 上記の行列変換を用いて画像の透視変換を行う
#----------
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(600,300))

cv2.imshow('Dst', dst)
cv2.imwrite('Dst.jpg', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()