#!/bin/bash

# Define the virtual environment directory
VENV_DIR="lacpvenv"
REPOSITORY_PATH="Deep_Q_Learning_Auto_IDS"
PYTHON_PATH="Deep_Q_Learning_Auto_IDS/src/model"
PYTHON_FILE="main.py"
JSON_PATH="Deep_Q_Learning_Auto_IDS/model_jsons/dql.json"

if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    python3 -m venv $VENV_DIR
    echo "$VENV_DIR virtual environment not found, creating one..."
    source "$VENV_DIR/bin/activate"
fi

# Check if requirements.txt exists
if [ -f "$REPOSITORY_PATH/requirements.txt" ]; then
    cd $REPOSITORY_PATH
    pip install -e .
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Make sure the repository folder is right"
    exit 1
fi

cd ..

if [ -f "$PYTHON_PATH/$PYTHON_FILE" ]; then 
    echo Running python script from $PYTHON_PATH...
    python3 $PYTHON_PATH/$PYTHON_FILE --config $JSON_PATH

else 
    echo "Python script not found!"
fi

echo "Environment setup complete."
