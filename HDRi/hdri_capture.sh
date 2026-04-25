#!/bin/bash
#

# Stop service blocking camera access
systemctl --user stop gvfs-gphoto2-volume-monitor.service

# Store images in SD card
gphoto2 --set-config-index capturetarget=1

if [ $? != 0 ]; then
    echo "No camera detected! Aborting operation"
    exit 1
fi

get_current_idx() {
    CURRENT=$(gphoto2 --get-config=shutterspeed | grep "Current" | cut -f2 -d' ')
    while read LINE; do
        PREFIX=$(echo $LINE | cut -f1 -d' ')
        if [ "$PREFIX" == "Choice:" ]; then
            VAL=$(echo $LINE | cut -f3 -d' ')
            if [ $VAL == $CURRENT ]; then
                IDX=$(echo $LINE | cut -f2 -d' ')
                echo $IDX
            fi
        fi
    done < <(gphoto2 --get-config=shutterspeed)
}

shoot() {
    gphoto2 --capture-image-and-download --hook-script hook.sh
}

CURRENT_IDX=$(get_current_idx)
MAX_VAL=52
MIN_VAL=1
# Exposure difference between shots - Canon steps = 1/3 EV (f-stop)
STEPS=3
# Number of shots over and under-exposed
#NSHOTS=5
OVERSHOTS=5
UNDERSHOTS=9

export FOLDER_NAME=capture_$(date '+%y%m%d_%H%M%S')
mkdir $FOLDER_NAME

# Correctly exposed shoot
shoot

# for i in $(seq 1 1 $NSHOTS); do
#     # Under-exposed (higher speed)
#     ss=$(expr $CURRENT_IDX + $i \* $STEPS)
#     if [ $ss -gt $MIN_VAL ]; then
#         gphoto2 --set-config-index shutterspeed=$ss
#         shoot
#     fi
#     # Over-exposed (lower speed)
#     ss=$(expr $CURRENT_IDX - $i \* $STEPS)
#     if [ $ss -lt $MAX_VAL ]; then
#         gphoto2 --set-config-index shutterspeed=$ss
#         shoot
#     fi
# done

for i in $(seq 1 1 $OVERSHOTS); do
    # Over-exposed (lower speed)
    ss=$(expr $CURRENT_IDX - $i \* $STEPS)
    if [ $ss -lt $MAX_VAL ]; then
        gphoto2 --set-config-index shutterspeed=$ss
        shoot
    fi
done

for i in $(seq 1 1 $UNDERSHOTS); do
    # Under-exposed (higher speed)
    ss=$(expr $CURRENT_IDX + $i \* $STEPS)
    if [ $ss -gt $MIN_VAL ]; then
        gphoto2 --set-config-index shutterspeed=$ss
        shoot
    fi
done

# HDR generation
luminance-hdr-cli --align AIS --save ${FOLDER_NAME}_CR2.exr $FOLDER_NAME/CR2/*.CR2
luminance-hdr-cli --align AIS --save ${FOLDER_NAME}_JPG.exr $FOLDER_NAME/JPG/*.JPG
