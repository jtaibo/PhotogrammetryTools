#!/bin/bash
#

case "$ACTION" in
    "download")
        EXT=$(echo $ARGUMENT | cut -f2 -d'.')
        if [ ! -d "$FOLDER_NAME/$EXT" ]; then
            mkdir $FOLDER_NAME/$EXT
        fi
        mv $ARGUMENT $FOLDER_NAME/$EXT
        ;;
esac
