#! /usr/bin/env bash

${PRODUCTION:=true}
VER=3

sudo apt update
sudo apt install ffmpeg python${VER}-pip -y
python$VER -m venv venv
source venv/bin/activate
export VIRTUAL_ENV
python$VER -m pip install --upgrade pip
pip install -r faster/gha/requirements.txt
pip install faster-whisper
python3 faster/gha/app.py $HOME/default.m4a
cat transcription.txt

