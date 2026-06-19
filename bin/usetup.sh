#! /usr/bin/env bash

FRAMEWORKS="flask gradio streamlit"
FRAMEWORK="$1"

main() {
	rootdir=$(pwd)
	if test -z $VIRTUAL_ENV; then
	cd
	test -d venv || python3 -m venv venv
	source venv/bin/activate
	export VIRTUAL_ENV
	fi
	cd $rootdir/$FRAMEWORK
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

main

