#! /usr/bin/env bash

AUDIO_FILE="$1"
DEVICE=cpu
odir=$HOME/transcriptions
VER=3.12

if ! command -v python$VER &>/dev/null; then
	brew install python@$VER
fi
mkdir -p $odir
python$VER -m venv venv
source venv/bin/activate
export VIRTUAL_ENV
python$VER -m pip install --upgrade pip
brew install ffmpeg-full
pip install -r requirements.txt
if test -z $HF_TOKEN; then
	echo "exportati HF_TOKEN"
else
	whispermlx "$AUDIO_FILE" --device $DEVICE --hf_token $HF_TOKEN --language ro --model large-v3 --output_dir $odir --output_format txt
	ofile=$odir/$(basename $AUDIO_FILE | sed -e "s/\.m4a//g").txt
	echo "=== Transcription"
	cat $ofile
	echo "==="
fi

