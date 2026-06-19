#! /usr/bin/env bash

PROJECT="at_1"
${PRODUCTION:=true}
VER=3

sudo apt update
sudo apt install ffmpeg python${VER}-pip -y
python$VER -m venv venv
source venv/bin/activate
python$VER -m pip install --upgrade pip
pip install -r requirements.txt
pip install faster-whisper
python3 app.py $HOME/default.m4a
cat transcription.txt

