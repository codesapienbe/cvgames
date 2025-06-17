# Module Runner Script

This script allows you to run different Python modules in the EHB project with simple commands from your current working directory.

## Setup

To set up the aliases, source the script in your terminal session:

```bash
source run.sh
```

You can add this line to your `~/.bashrc` or `~/.zshrc` file to make the aliases available in all terminal sessions:

```bash
# Add to ~/.bashrc or ~/.zshrc
source /path/to/run.sh
```

## Usage

After sourcing the script, you can run any module with a simple command:

```bash
# Format: mp-<module_name>
mp-body     # Runs mediapipe-body
mp-hands    # Runs mediapipe-hands
mp-face     # Runs mediapipe-face
```

You can also pass additional arguments:

```bash
mp-body --debug
```

## Running Without Aliases

If you prefer not to set up aliases, you can run modules directly with the script:

```bash
./run.sh mediapipe-body
```

## Available Modules

To see a list of available modules:

```bash
./run.sh
```

## How It Works

The script:

1. Uses your current working directory to find modules
2. Creates aliases for each directory found
3. Automatically converts "mediapipe-xxx" to "mp-xxx" for shorter commands
4. Runs the main.py file in the specified module directory
5. Passes any additional arguments to the Python script

## Adding New Modules

When you add a new module to the project:

1. Create a directory for it (e.g., `mediapipe-newfeature`)
2. Add a `main.py` file as the entry point
3. Re-source the script to update the aliases: `source run.sh`

The new module will be available with the command `mp-newfeature`.
