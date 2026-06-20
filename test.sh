#! /usr/bin/env bash

PLATFORM=$(uname)
echo "platform: $PLATFORM"
script=bin/test.sh
[ $PLATFORM = linux ] && script=bin/utest.sh
echo -n "Introduceti framework-ul (cli/flask/gradio/streamlit): "
read FRAMEWORK
echo -n "Introduceti tool-ul (faster/mlx): "
read TOOL
export FRAMEWORK TOOL

if [ $TOOL == mlx ]; then
	if [ $FRAMEWORK = cli ]; then
		echo -n "Introduceti calea catre fisierul audio: "
		read AF
		bash mlx/$FRAMEWORK/test.sh $AF
	else
		bash $script
	fi
else
	if [ $FRAMEWORK = cli ]; then
		echo -n "Introduceti calea catre fisierul audio: "
		read AF
		bash $script $AF
	else
		bash $script
	fi
fi

