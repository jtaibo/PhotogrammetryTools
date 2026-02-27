#!/bin/bash
#

sudo apt install exiftool ffmpeg imagemagick

if [ ! -d venv ]; then
  python -m venv venv
fi

source venv/bin/activate

python -m pip install -U pyexiftool
python -m pip install -U opencv-python
