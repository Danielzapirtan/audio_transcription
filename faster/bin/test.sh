#! /usr/bin/env bash

PROJECT="at_1"
DIR=$(pwd)
${PRODUCTION:=true}
VER=3.12

brew update
brew install python@$VER
brew install ffmpeg-full
python$VER -m venv venv
source venv/bin/activate
python$VER -m pip install --upgrade pip
pip install -r requirements.txt
pip install faster-whisper
python$VER app.py $HOME/default.m4a
cat transcription.txt

