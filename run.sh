#!/bin/bash

# Directory where the modules are located - use current working directory
PROJECT_DIR=${PWD}

# Function to run a module
run_module() {
    MODULE_NAME=$1
    shift  # Remove the first argument, keeping any additional arguments
    
    if [ -d "$PROJECT_DIR/$MODULE_NAME" ]; then
        echo "Running $MODULE_NAME..."
        
        # Activate virtual environment if it exists
        if [ -d "$PROJECT_DIR/.venv" ]; then
            echo "Activating virtual environment..."
            source "$PROJECT_DIR/.venv/bin/activate"
        else
            echo "Warning: No .venv directory found. Running with system Python."
        fi
        
        # Use Python from PATH - more portable
        python3 "$PROJECT_DIR/$MODULE_NAME/main.py" "$@"
        
        # Deactivate virtual environment if it was activated
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate
        fi
    else
        echo "Module $MODULE_NAME not found in $PROJECT_DIR"
        echo "Available modules:"
        find "$PROJECT_DIR" -maxdepth 1 -type d -not -path "$PROJECT_DIR" -exec basename {} \; | sort
    fi
}

# Check if being sourced or executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <module-name> [additional arguments]"
        echo "Available modules:"
        find "$PROJECT_DIR" -maxdepth 1 -type d -not -path "$PROJECT_DIR" -exec basename {} \; | sort
        exit 1
    fi
    
    run_module "$@"
else
    # Script is being sourced - set up aliases
    echo "Setting up module aliases..."
    
    # Find all directories in the project directory (potential modules)
    for MODULE_DIR in "$PROJECT_DIR"/*/ ; do
        if [ -d "$MODULE_DIR" ]; then
            MODULE_NAME=$(basename "$MODULE_DIR")
            ALIAS_NAME="mp-$(echo "$MODULE_NAME" | sed 's/mediapipe-//')"
            
            # Create an alias for the module that uses the full path
            alias "$ALIAS_NAME"="run_module $MODULE_NAME"
            echo "Created alias: $ALIAS_NAME -> $MODULE_NAME"
        fi
    done
    
    echo "Module aliases are set up. Use mp-<module> to run a module."
    echo "For example: mp-body to run mediapipe-body"
fi