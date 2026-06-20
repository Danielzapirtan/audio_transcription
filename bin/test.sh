#! /usr/bin/env bash

ARG="$1"
${PRODUCTION:=true}
VER=3.12

FRAMEWORKS="gha flask gradio streamlit"
TOOLS="faster mlx"

main() {
	rootdir=$(pwd)
	AF="$rootdir/$ARG"
	brew install python@$VER
	if test -z $VIRTUAL_ENV; then
	test -d venv || python$VER -m venv venv
	source venv/bin/activate
	export VIRTUAL_ENV
	fi
	cd $TOOL/$FRAMEWORK
	brew install ffmpeg
	pip install -r requirements.txt
	if [ $FRAMEWORK = streamlit ]; then
		pkill -kill streamlit
		streamlit run app.py
	elif [ $FRAMEWORK = gha ]; then
		python$VER app.py "$AF"
	else
		python$VER app.py
	fi
}

echo "$FRAMEWORKS" | grep -q $FRAMEWORK || exit 2
echo "$TOOLS" | grep -q $TOOL || exit 3

main

