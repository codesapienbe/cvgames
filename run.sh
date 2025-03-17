#!/bin/bash

# Directory where the modules are located - use current working directory
PROJECT_DIR=${PWD}

# Function to run a module
run_module() {
    
    MODULE_NAME=$1
    shift  # Remove the first argument, keeping any additional arguments
    
    if [ -d "$PROJECT_DIR/$MODULE_NAME" ]; then
        echo "Running $MODULE_NAME..."
        
        # Check for module-specific virtual environment
        MODULE_VENV="$PROJECT_DIR/$MODULE_NAME/.venv"
        
        # Create module-specific virtual environment if it doesn't exist
        if [ ! -d "$MODULE_VENV" ]; then
            echo "Creating virtual environment for $MODULE_NAME..."
            python3 -m venv "$MODULE_VENV"
            
            # Install module-specific requirements if they exist
            if [ -f "$PROJECT_DIR/$MODULE_NAME/requirements.txt" ]; then
                echo "Installing requirements for $MODULE_NAME..."
                "$MODULE_VENV/bin/pip" install -r "$PROJECT_DIR/$MODULE_NAME/requirements.txt"
            else
                echo "No requirements.txt found for $MODULE_NAME."
            fi
        fi
        
        # Activate module-specific virtual environment
        echo "Activating virtual environment for $MODULE_NAME..."
        source "$MODULE_VENV/bin/activate"
        
        # Use Python from PATH - more portable
        python3 "$PROJECT_DIR/$MODULE_NAME/main.py" "$@"
        
        # Deactivate virtual environment
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate
        fi
    else
        echo "Module $MODULE_NAME not found in $PROJECT_DIR"
        echo "Available modules:"
        find "$PROJECT_DIR" -maxdepth 1 -type d -not -path "$PROJECT_DIR" -not -path "$PROJECT_DIR/.git" -exec basename {} \; | sort
    fi
}

# Function to initialize virtual environment for a module
initialize_venv() {
    MODULE_NAME=$1
    MODULE_VENV="$PROJECT_DIR/$MODULE_NAME/.venv"
    
    if [ ! -d "$MODULE_VENV" ]; then
        echo "Creating virtual environment for $MODULE_NAME..."
        python3 -m venv "$MODULE_VENV"
        
        # Install module-specific requirements if they exist
        if [ -f "$PROJECT_DIR/$MODULE_NAME/requirements.txt" ]; then
            echo "Installing requirements for $MODULE_NAME..."
            "$MODULE_VENV/bin/pip" install -r "$PROJECT_DIR/$MODULE_NAME/requirements.txt"
        else
            echo "No requirements.txt found for $MODULE_NAME."
        fi
    fi
}

# Check if being sourced or executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <module-name> [additional arguments]"
        echo "Available modules:"
        find "$PROJECT_DIR" -maxdepth 1 -type d -not -path "$PROJECT_DIR" -not -path "$PROJECT_DIR/.git" -exec basename {} \; | sort
        exit 1
    fi
    
    run_module "$@"
else
    # Script is being sourced - set up aliases
    echo "Setting up module aliases..."
    
    # Find all directories in the project directory (potential modules)
    for MODULE_DIR in "$PROJECT_DIR"/*/ ; do
        if [ -d "$MODULE_DIR" ] && [ "$MODULE_DIR" != "$PROJECT_DIR/.git/" ]; then
            MODULE_NAME=$(basename "$MODULE_DIR")
            
            # Initialize virtual environment for this module
            initialize_venv "$MODULE_NAME"
            
            # Create an alias for the module that uses the full path
            alias "$MODULE_NAME"="run_module $MODULE_NAME"
            echo "Created alias: $MODULE_NAME"
        fi
    done
    
    echo "Module aliases are set up. Use <module> to run a module."
    echo "For example: use the module name directly as the command"
fi