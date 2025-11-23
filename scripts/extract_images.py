import glob
import os
import cv2
import argparse
import json
import csv
import statistics
import numpy as np
import datetime
import sys
from timeit import default_timer as timer
import exiftool


# NOTE: In this example, "U:/ffmpeg/bin" path contains ffmpeg and exiftool executables
os.environ["PATH"] = "U:/ffmpeg/bin;" + os.environ["PATH"]

parser = argparse.ArgumentParser("extract_images")
parser.add_argument("input_video", help="Path for the video to split into frames", default="")
parser.add_argument("--exif_ref", help="Path to the image with the EXIF metadata to populate the frames extracted from the video")
parser.add_argument("--folder", help="Path to folder where to place the extracted frames")
parser.add_argument("--framePrefix", help="Frame file name prefix")
parser.add_argument("--frameScale", help="Scaling factor for extracted frames", default=1)

# TO-DO
# Keep all frames (raw, filtered)
parser.add_argument("--keep_all", help="Keep all frames extracted from video or filtered ones only", default = False)

# Step selection (extract, filter, exif)
parser.add_argument("--no_extract", help="Skip frames extraction (frames are already extracted)", action=argparse.BooleanOptionalAction, default = False)
parser.add_argument("--no_filter", help="Extract frames only, do not filter them", action=argparse.BooleanOptionalAction, default = False)
# EXIF are already skipped when no reference file is given

# EXIF from file or from parameters (make, model, sensor_width)

parser.add_argument("--report_stats", help="Report statistics (sharpness)", action=argparse.BooleanOptionalAction, default = False)

parser.add_argument("--sharpness_threshold", type=float, default = 55)
parser.add_argument("--diff_threshold", type=float, default = 0)
parser.add_argument("--flow_threshold", type=float, default = 30)

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
#   Extract EXIF metadata from video and copy it to extracted frames
###############################################################################
def copyEXIFFromVideo(input_file, output_name, frames_folder):
    # EXPERIMENTAL
    pass


###############################################################################
#   Extract frames to a directory
#
#   TO-DO: expose parameters (start time, duration, fps, resolution, ...)
###############################################################################
def extractFrames(input_file, output_name, frames_folder, scale):
   
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

    SCALE=scale

    cmd = "ffmpeg -ss " + str(START_TIME) + " -t " + str(DURATION) + " -i \"" + input_file + "\" -r " + str(FRAME_STEP) + " -vf scale=\"iw*" + str(SCALE) + ":ih*" + str(SCALE) +"\" " + frames_folder + "/" + output_name + "%04d." + EXTENSION
    os.system(cmd)
    #print(cmd)

    copyEXIFFromVideo(input_file, output_name, frames_folder)

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

#    for filepath in frames_list:
#        print("Adding EXIF metadata to " + filepath)
#        cmd = exiftool_bin + " -overwrite_original -TagsFromFile \"" + metadata_source + "\" \"" + filepath + "\""
        #print(cmd)
#        os.system(cmd)
        #print(filepath)


    with exiftool.ExifToolHelper() as et:

        tags = et.get_tags( exif_ref, tags = ["make", "model", "focalLength"])

        make = tags[0]["EXIF:Make"]
        model = tags[0]["EXIF:Model"]
        focal_length = tags[0]["EXIF:FocalLength"]
        #serial_number = tags[0]["EXIF:SerialNumber"]

        et.set_tags( frames_list,
                    tags={"make": make,
                          "model": model,
                          "focalLength": focal_length},
                          #"serialNumber": serial_number},
                    params=["-overwrite_original"]
        )


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
def filterImages(frames_list, sharpness_threshold, diff_threshold, flow_threshold):
    # The filtering thresholds can come from CLI arguments or from statistics
    # computed among all extracted frames to keep the better ones
    # We could also limit the number of frames to use so the script can select
    # the best n frames
    previous = None
    for filepath in frames_list:
        score = computeSharpness(filepath)
        #print(filepath + " -> " + str(score))
        if score < sharpness_threshold:
            print("Sharpness level", score, "lower than threshold", sharpness_threshold, "Discarding frame", filepath)
            os.remove(filepath)
        else:
            # Check difference with previous frame
            if previous:
                diff, flow = computeDifference(previous, filepath)
                #print(previous, "->", filepath, ":", diff, flow)
                if diff < diff_threshold or flow < flow_threshold:
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

    print("Starting frame extraction and processing - ", datetime.datetime.now().isoformat())
    print("COMMAND LINE:", sys.argv)
    start_time = timer()

    if not args.folder:
        frames_folder = os.path.basename(args.input_video)
    else:
        frames_folder = args.folder

    if not args.framePrefix:
        frame_prefix = frames_folder + "_"
    else:
        frame_prefix = args.framePrefix

    if args.no_extract:
        print("Skipping extraction stage")
    else:
        print("Extracting frames...")
        extractFrames(args.input_video, frame_prefix, frames_folder, args.frameScale)
    #removeBlurredImages(getFramesInFolder(frames_folder))
    #removeSimilarImages(getFramesInFolder(frames_folder))

    if args.report_stats:
        stats_filename = frame_prefix + 'stats'
        print("Dumping stats to", stats_filename)
        sharpness = getSharpness(getFramesInFolder(frames_folder))
        sh_mean = statistics.mean(sharpness.values())
        sh_min = min(sharpness.values())
        sh_max = max(sharpness.values())
        sh_stdev = statistics.stdev(sharpness.values())
        #print("MEAN:", sh_mean, "MIN:", sh_min, "MAX:", sh_max, "STDEV:", sh_stdev)
        # JSON
        with open(stats_filename + ".json", 'w') as fp:
            json.dump(sharpness, fp)
        # CSV
        with open(stats_filename + ".csv", 'w', newline='') as fp:
            csvwriter = csv.writer(fp, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerows(sharpness.items())

    # Filter images (considering blurriness and difference with previous image)
    if args.no_filter:
        print("Skipping filtering stage")
    else:
        print("Filtering images...")
        filterImages(getFramesInFolder(frames_folder), args.sharpness_threshold, args.diff_threshold, args.flow_threshold)

    if args.exif_ref:
        print("Adding EXIF metadata...")
        addExifInfo(getFramesInFolder(frames_folder), args.exif_ref)

    print("Finished frame extraction and processing - ", datetime.datetime.now().isoformat())
    end_time = timer()
    print("Elapsed time:", end_time-start_time, "seconds")

main()
