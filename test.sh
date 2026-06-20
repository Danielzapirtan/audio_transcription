#! /bin/bash

PLATFORM=$(uname)
echo "platform: $PLATFORM"
if [ $PLATFORM = linux ]; then
	MYSCRIPT=utest.sh
else
	MYSCRIPT=test.sh
fi
echo -n "Introduceti framework-ul (cli/flask/gradio/streamlit): "
read FRAMEWORK
echo -n "Introduceti tool-ul (faster/mlx): "
read TOOL
export FRAMEWORK TOOL

if [ $TOOL = mlx ]; then
	if [ $FRAMEWORK = cli ]; then
		echo -n "Introduceti calea catre fisierul audio: "
		read AF
		bash mlx/$FRAMEWORK/test.sh $AF
	else
		bash bin/$MYSCRIPT
	fi
else
	if [ $FRAMEWORK = cli ]; then
		echo -n "Introduceti calea catre fisierul audio: "
		read AF
		bash bin/$MYSCRIPT $AF
	else
		bash bin/$MYSCRIPT
	fi
fi

