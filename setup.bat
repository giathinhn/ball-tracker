@echo off
if not exist "model" (
    mkdir "model"
) else (
    echo 'model' directory already exists.
)

echo Downloading football-ball-detection.pt...
python -m gdown -O model/football-ball-detection.pt "https://drive.google.com/file/d/15akhXwgUa5aCxtqk0Ca67ih_-mR5Yf89/view?usp=sharing"

echo Downloading football-player-detection.pt...
python -m gdown -O model/football-player-detection.pt "https://drive.google.com/file/d/18HIhyYQZ7EZHFrfxks8CLjY4TFud31-_/view?usp=sharing"

echo Downloading football-pitch-detection.pt...
python -m gdown -O model/football-pitch-detection.pt "https://drive.google.com/file/d/1Zy8Gk_7AneAvRCar8u1FKsNoXKGr1OO7/view?usp=sharing"

echo Setup completed successfully!
