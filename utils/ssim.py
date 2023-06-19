import os
import shutil

import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import structural_similarity as ssim


def compute_ssim(cam_img_1, cam_img_2):
    grayA = cv2.imread(cam_img_1, cv2.IMREAD_GRAYSCALE)
    grayB = cv2.imread(cam_img_2, cv2.IMREAD_GRAYSCALE)
    (score, __) = structural_similarity(grayA, grayB, full=True)
    return score


def compute_ssim_full(file_pathname, files, outFile, in_sim, fi):
    grayA = cv2.imread(file_pathname+"/"+files[fi], cv2.IMREAD_GRAYSCALE)
    grayB = cv2.imread(file_pathname+"/"+files[fi+1], cv2.IMREAD_GRAYSCALE)
    imgColor = cv2.imread(file_pathname+"/"+files[fi])
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    print(fi, score)
    if (score < in_sim or True):
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[1] if imutils.is_cv2() else cnts[0]
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imgColor, (x, y), (x+w, y+h), (0, 0, 255), 2)
        outFile.writelines(files[fi+1]+","+str(score)+"\r\n")


if __name__ == '__main__':
    file_pathname = r"/home2/anonymous/lim3d-2/data/sequences/00/image_3"
    files = os.listdir(file_pathname)

    files.sort()
    cnt = len(files)
    in_subsim = 0.12
    outFile = open('out.txt', mode='w')
    outFile.writelines("filename,score\r\n")
    in_sim = 0.18

    for fi in range(cnt-1):
        compute_ssim_full(file_pathname, files, outFile, in_sim, fi)

    outFile.close()
