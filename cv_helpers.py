import cv2
cv = cv2
import skvideo.io
import numpy as np
import os
import scipy

"""
basic helper functions
"""

def split_path(path):
    root, file = os.path.split(path)
    name, ext = os.path.splitext(file)
    return root, name, ext

def strip_extension(name):
    reverse_ind = name[::-1].find(".")
    if reverse_ind != -1:
        name = name[:-reverse_ind-1]
    return name

def readvid(file, maxframes=None):
    print("Reading", file)
    cap = cv.VideoCapture(file)
    vidframes = []
    count = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        vidframes.append(frame)
        if maxframes is not None:
            count += 1
            if count >= maxframes:
                break
    return vidframes

def writevid(vid, name, flipchannels=True):
    """
    name: filename to save under
    flipchannels: bool, true if video is in BGR
    """
    name = strip_extension(name)+".mp4"
    print("writing vid", name, "...")
    vid = np.array(vid)
    if flipchannels:
        vid = vid[...,::-1]
    skvideo.io.vwrite(name, vid, 
        outputdict={"-pix_fmt": "yuv420p"},
        backend='ffmpeg')
    # fourcc = cv.CV_FOURCC(*"mp4v")
    # writer = cv.VideoWriter()
    # writer.open(name, fourcc, 20.0, vid[0].shape[:2], True)
    # for frame in vid:
    #     writer.write(frame)
    # writer.release()

def showim(img, name="window", ms=1000):
    """
    show image with a good wait time
    """
    cv2.imshow(name, img)
    cv.moveWindow(name, 0, 0)
    cv2.waitKey(ms)
    cv.destroyWindow(name)
    cv.waitKey(1)

def showvid(vid, name="window", ms=25):
    """
    show vid, press a key to cancel
    """
    for frame in vid:
        cv.imshow(name, frame)
        cv.moveWindow(name, 0, 0)
        if cv.waitKey(ms) != -1:
            break
    try:
        cv.destroyWindow(name)
        cv.waitKey(1)
    except:
        pass

def annotate_vid(vid, preds, trues, categorical):
    for i, frame in enumerate(vid):
        xloc = 5
        for j in range(6):
            if categorical:
                pred = np.argmax(preds[i][j])
            else:
                pred = round(preds[i][j])
            true = trues[i][j]
            if pred == true:
                c = (0,255,0)
            else:
                c = (0,0,255)
            cv.putText(vid[i], str(pred), (xloc, 20), cv.FONT_HERSHEY_PLAIN, fontScale=2, 
                    color=c, thickness=2)
            cv.putText(vid[i], str(true), (xloc, 40), cv.FONT_HERSHEY_PLAIN, fontScale=2, 
                    color=(255,255,255), thickness=2)
            xloc += 15

def euc_dist(p1, p2s):
    return np.sqrt(p1 ** 2 + p2s ** 2)

def order_points(pts):
    """
    order contour points, from 
    https://github.com/jrosebr1/imutils/blob/224d591b4c3a9efc855e2e8eabc9c55199c696c3/imutils/perspective.py
    """
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = scipy.spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype=np.int16)
