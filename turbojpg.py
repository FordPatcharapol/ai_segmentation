import cv2
import numpy as np
import exifread
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
import time

# using default library installation
jpeg = TurboJPEG()

path = './data/test-case.jpg'

# decoding input.jpg to BGR array
s = time.perf_counter()

in_file = open(path, 'rb')
bgr_array = jpeg.decode(in_file.read())
in_file.close()

elapsed = time.perf_counter() - s
print(f"{elapsed} seconds.")

s = time.perf_counter()

bgr_array = cv2.imread(path)
elapsed = time.perf_counter() - s

print(f"{elapsed} seconds.")
# cv2.imshow('bgr_array', bgr_array)
# cv2.waitKey(0)