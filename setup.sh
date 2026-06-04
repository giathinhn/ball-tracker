#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if 'model' directory does not exist and then create it
if [[ ! -e $DIR/model ]]; then
    mkdir "$DIR/model"
else
    echo "'model' directory already exists."
fi

# download the models
gdown -O "$DIR/model/football-ball-detection.pt" "https://drive.google.com/file/d/15akhXwgUa5aCxtqk0Ca67ih_-mR5Yf89/view?usp=sharing"
gdown -O "$DIR/model/football-player-detection.pt" "https://drive.google.com/file/d/18HIhyYQZ7EZHFrfxks8CLjY4TFud31-_/view?usp=sharing"
gdown -O "$DIR/model/football-pitch-detection.pt" "https://drive.google.com/file/d/1Zy8Gk_7AneAvRCar8u1FKsNoXKGr1OO7/view?usp=sharing"
