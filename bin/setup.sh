#! /usr/bin/env bash

${PRODUCTION:=true}
VER=3.12

FRAMEWORKS="flask gradio streamlit"
FRAMEWORK="$1"

main() {
	rootdir=$(pwd)
	brew install python@$VER
	if test -z $VIRTUAL_ENV; then
	cd
	test -d venv || python$VER -m venv venv
	source venv/bin/activate
	export VIRTUAL_ENV
	fi
	cd $rootdir/$FRAMEWORK
	brew install ffmpeg
	pip install -r requirements.txt
	if [ $FRAMEWORK = streamlit ]; then
		pkill -kill streamlit
		streamlit run app.py
	else
		python$VER app.py
	fi
}

echo "$FRAMEWORKS" | grep -q $FRAMEWORK || exit 2
test -n "$VIRTUAL_ENV" || exit 3

main

