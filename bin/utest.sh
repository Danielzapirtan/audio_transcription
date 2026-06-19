#! /usr/bin/env bash

FRAMEWORKS="cli colab flask gha gradio streamlit"
FRAMEWORK="$1"

TOOLS="faster mlx"
TOOL="$2"

main() {
	rootdir=$(pwd)
	if test -z $VIRTUAL_ENV; then
	test -d venv || python3 -m venv venv
	source venv/bin/activate
	export VIRTUAL_ENV
	fi
	cd $TOOL/$FRAMEWORK
	apt install ffmpeg
	pip install -r requirements.txt
	if [ $FRAMEWORK = streamlit ]; then
		pkill -kill streamlit
		streamlit run app.py
	else
		python3 app.py
	fi
}

echo "$FRAMEWORKS" | grep -q $FRAMEWORK || exit 2
echo "$TOOLS" | grep -q $TOOL || exit 3

main

