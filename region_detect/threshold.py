import cv2 as cv
from numpy import uint8

RGB_WHITE = (255, 255, 255)


def to_numpy(img):
    return img.clone().detach().cpu().numpy()


def binarize(image, threshold):
    return (image > threshold).astype(uint8)


def mer(image):
    return cv.boundingRect(image)
