#! /usr/bin/env bash

ARG=$1
FRAMEWORKS="cli gha flask gradio streamlit"
TOOLS="faster mlx"

main() {
	rootdir=$(pwd)
	AF="$rootdir/$ARG"
	if test -z $VIRTUAL_ENV; then
	test -d venv || python3 -m venv venv
	source venv/bin/activate
	export VIRTUAL_ENV
	fi
	cd $TOOL/$FRAMEWORK
	sudo apt install ffmpeg
	pip install -r requirements.txt
	if [ $FRAMEWORK = streamlit ]; then
		pkill -kill streamlit
		streamlit run app.py
	elif [ $FRAMEWORK = gha ]; then
		python3 app.py "$AF"
	else
		python3 app.py
	fi
}

echo "$FRAMEWORKS" | grep -q $FRAMEWORK || exit 2
echo "$TOOLS" | grep -q $TOOL || exit 3

main

