import glob
import os
import cv2
import argparse
#import json
#import statistics
import numpy as np


# NOTE: In this example, "U:/ffmpeg/bin" path contains ffmpeg and exiftool executables
os.environ["PATH"] = "U:/ffmpeg/bin;" + os.environ["PATH"]

parser = argparse.ArgumentParser("extract_images")
parser.add_argument("input_video", help="Path for the video to split into frames", default="")
parser.add_argument("exif_ref", help="Path to the image with the EXIF metadata to populate the frames extracted from the video", default="")
parser.add_argument("--folder", help="Path to folder where to place the extracted frames")
parser.add_argument("--framePrefix", help="Frame file name prefix")

parser.add_argument("--sharpness_threshold", default = 55)
parser.add_argument("--diff_threshold", default = 0)
parser.add_argument("--flow_threshold", default = 30)

args = parser.parse_args()


###############################################################################
#   Return video parameters
###############################################################################
def getVideoInfo(video_path):
    video = cv2.VideoCapture(video_path)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frameCount / fps
    video.release()
    return {"width": width, "height":height, "frameCount": frameCount, "fps": fps, "duration": duration}


###############################################################################
#   Extract frames to a directory
#
#   TO-DO: expose parameters (start time, duration, fps, resolution, ...)
###############################################################################
def extractFrames(input_file, output_name, frames_folder):
   
    # Create a new directory to avoid messing up too much
    if os.path.isdir(frames_folder):
        print("Frames have already been extracted. Stopping now")
        os.abort()

    os.mkdir(frames_folder)

    vinfo = getVideoInfo(input_file)

    START_TIME=0
    DURATION=vinfo["duration"]

    # Get n frames per second
    FRAME_STEP=vinfo["fps"]
    EXTENSION="jpg"

    cmd = "ffmpeg -ss " + str(START_TIME) + " -t " + str(DURATION) + " -i \"" + input_file + "\" -r " + str(FRAME_STEP) + " " + frames_folder + "/" + output_name + "%04d." + EXTENSION
    os.system(cmd)
    #print(cmd)

###############################################################################
#
###############################################################################
def getFramesInFolder(frames_folder):
    return glob.iglob(frames_folder+'/*.jpg')

###############################################################################
# 
###############################################################################
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

###############################################################################
#   Compute amount of sharpness/blurriness in an image
#   Returns a numerical value of sharpness (variance of laplacian)
#
# Prerequisites: opencv-python, imutils
# https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
###############################################################################
def computeSharpness(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

###############################################################################
#   Copy EXIF metadata to all frames in a list from a reference image
###############################################################################
def addExifInfo(frames_list, exif_ref):
    exiftool_bin = "exiftool.exe"
    metadata_source = exif_ref

    for filepath in frames_list:
        print("Adding EXIF metadata to " + filepath)
        cmd = exiftool_bin + " -overwrite_original -TagsFromFile " + metadata_source + " " + filepath
        #print(cmd)
        os.system(cmd)
        #print(filepath)

###############################################################################
# 
###############################################################################
def computeDifference(frame1, frame2):

    imgA = cv2.imread(frame1)
    imgB = cv2.imread(frame2)

    imgAgray = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    imgBgray = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(imgAgray, imgBgray)

    flow = cv2.calcOpticalFlowFarneback(imgAgray, imgBgray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow3 = np.zeros_like(imgA)
    flow3[:,:,0]=flow[:,:,0]
    flow3[:,:,1]=flow[:,:,1]

    diff_avg = cv2.reduce(diff, 0, cv2.REDUCE_AVG).mean()
    #print("diff_avg:", diff_avg)
    flow_avg = cv2.reduce(flow3, 0, cv2.REDUCE_AVG).mean()
    #print("flow_avg:", flow_avg)

    return diff_avg, flow_avg


###############################################################################
#   Get sharpness levels
###############################################################################
def getSharpness(frames_list):
    sharpness_levels = {}
    for filepath in frames_list:
        score = computeSharpness(filepath)
        sharpness_levels[filepath] = score
    return sharpness_levels

###############################################################################
# 
###############################################################################
def filterImages(frames_list):
    # The filtering thresholds can come from CLI arguments or from statistics
    # computed among all extracted frames to keep the better ones
    # We could also limit the number of frames to use so the script can select
    # the best n frames
    previous = None
    for filepath in frames_list:
        score = computeSharpness(filepath)
        #print(filepath + " -> " + str(score))
        if score < args.sharpness_threshold:
            print("Sharpness level", score, "lower than threshold", args.sharpness_threshold, "Discarding frame", filepath)
            os.remove(filepath)
        else:
            # Check difference with previous frame
            if previous:
                diff, flow = computeDifference(previous, filepath)
                #print(previous, "->", filepath, ":", diff, flow)
                if diff < args.diff_threshold or flow < args.flow_threshold:
                    print("Low difference. Removing frame ", filepath)
                    os.remove(filepath)
                else:
                    print(">>>>>> ENOUGH difference. Keeping frame", filepath)
                    previous = filepath
            else:
                previous = filepath

###############################################################################
#   MAIN
###############################################################################
def main():

    if not args.folder:
        frames_folder = os.path.basename(args.input_file)
    else:
        frames_folder = args.folder

    if not args.framePrefix:
        frame_prefix = frames_folder + "_"
    else:
        frame_prefix = args.framePrefix

    extractFrames(args.input_video, frame_prefix, frames_folder)
    #removeBlurredImages(getFramesInFolder(frames_folder))
    #removeSimilarImages(getFramesInFolder(frames_folder))

    # sharpness = getSharpness(getFramesInFolder(frames_folder))
    # sh_mean = statistics.mean(sharpness.values())
    # sh_min = min(sharpness.values())
    # sh_max = max(sharpness.values())
    # sh_stdev = statistics.stdev(sharpness.values())
    # print("MEAN:", sh_mean, "MIN:", sh_min, "MAX:", sh_max, "STDEV:", sh_stdev)
    # with open('sharpness.json', 'w') as fp:
    #     json.dump(sharpness, fp)

    # Filter images (considering blurriness and difference with previous image)
    filterImages(getFramesInFolder(frames_folder))

    addExifInfo(getFramesInFolder(frames_folder), args.exif_ref)

main()
