#! /usr/bin/env bash

DIR=$(pwd)
VER=3.12

brew update
brew install python@$VER
brew install ffmpeg-full
python$VER -m venv venv
source venv/bin/activate
export VIRTUAL_ENV
python$VER -m pip install --upgrade pip
pip install -r faster/gha/requirements.txt
pip install faster-whisper
python$VER faster/gha/app.py samples/default.m4a
cat transcription.txt

