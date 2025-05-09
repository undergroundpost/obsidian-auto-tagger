# Obsidian Auto Tagger

A Python script that uses AI to automatically generate and add relevant tags to your Obsidian notes. The script scans for notes created or modified on the previous day, analyzes their content, and adds contextually appropriate tags to the frontmatter.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Key Features

- 🔍 Scans your Obsidian vault for notes created or modified yesterday
- 🤖 Uses AI (Ollama locally or OpenAI) to generate contextually relevant tags
- 📊 Analyzes existing tags in your vault for consistency
- 🏷️ Updates note frontmatter with new tags (without duplicating existing ones)
- ⏱️ Built-in rate limit handling and advanced retry mechanisms
- 📝 Comprehensive logging for troubleshooting

## Why Use This Script?

Maintaining a consistent tagging system in Obsidian can be challenging but is crucial for effective knowledge management. This script helps solve common tagging problems:

- **Consistency**: By analyzing your entire vault's existing tags, it maintains a cohesive tagging system
- **Reduced Cognitive Load**: No need to stop and think about appropriate tags while writing
- **Improved Discoverability**: Better tagging means your notes are easier to find later
- **Time-Saving**: Automatically processes new notes while you sleep
- **Knowledge Connections**: Good tags help reveal connections between seemingly unrelated notes

The script is especially useful as part of a daily note-taking workflow. Run it as a nightly cron job, and each morning your new notes will be properly tagged and ready for reference.

## Requirements

- Python 3.7+
- Obsidian vault with markdown files
- One of the following:
  - [Ollama](https://ollama.ai/) (free, runs locally)
  - OpenAI API key (paid, cloud-based)
- Required Python packages:
  - requests
  - python-dateutil
  - pyyaml
  - openai (optional, for OpenAI API)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/undergroundpost/obsidian-auto-tagger.git
cd obsidian-auto-tagger
```

2. Install the required packages:
```bash
pip install requests python-dateutil pyyaml
# If using OpenAI
pip install openai
```

3. Copy and modify the example config file:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your preferred settings
```

4. Make the script executable (on Unix-like systems):
```bash
chmod +x generate_tags.py
```

5. Place the prompt file in the same directory:
```bash
# Make sure generate_tags.md is in the same directory as the script
```

## Configuration

You can configure the script in three ways:

1. **Config file**: Create a `config.yaml` file in the same directory as the script
2. **Environment-specific locations**: The script will search in these locations (in order):
   - Script directory: `./config.yaml`
   - User config: `~/.config/generate_tags/config.yaml`
   - System-wide: `/etc/generate_tags/config.yaml`
3. **Command-line options**: Override settings for a single run

### Config File Example

```yaml
# Folder settings
INPUT_FOLDER: "/Users/username/Obsidian/Vault" 

# Exclude folders (list of folders to ignore when scanning)
EXCLUDE_FOLDERS:
  - "/Users/username/Obsidian/Vault/AI"            # AI-related folder
  - "/Users/username/Obsidian/Vault/Private"       # Private notes
  - "/Users/username/Obsidian/Vault/Templates"     # Templates folder

# LLM Provider settings
LLM_PROVIDER: "ollama"                             # Options: "ollama" or "openai"

# Ollama settings (used when LLM_PROVIDER is "ollama")
OLLAMA_MODEL: "gemma3:12b"                         # Model to use
OLLAMA_SERVER_ADDRESS: "http://localhost:11434"    # Ollama server address
OLLAMA_CONTEXT_WINDOW: 32000                       # Context window size

# OpenAI settings (used when LLM_PROVIDER is "openai")
OPENAI_API_KEY: ""                                 # Your OpenAI API key (required for OpenAI)
OPENAI_MODEL: "gpt-3.5-turbo"                      # OpenAI model to use (gpt-3.5-turbo is more widely available)
OPENAI_MAX_TOKENS: 4000                            # Maximum tokens for responses
```

## Prompt File

The script requires a prompt file named `generate_tags.md` in the same directory as the script. This file contains instructions for the LLM on how to generate tags. A template is provided in the repository.

## Usage

### Basic usage

```bash
./generate_tags.py
```

### Specify a different date

```bash
./generate_tags.py --date 2025-05-01
```

### Override configuration

```bash
./generate_tags.py --input "/path/to/vault" --model "llama3:8b" --server "http://192.168.1.100:11434"
```

### Enable debug logging

```bash
./generate_tags.py --debug
```

## Command-line Options

| Option         | Description                                     |
|----------------|-------------------------------------------------|
| `--date`       | Override date to check (YYYY-MM-DD)             |
| `--debug`      | Enable detailed debug logging                   |
| `--input`      | Override input folder                           |
| `--exclude`    | Override exclude folders (can be used multiple times) |
| `--provider`   | Set LLM provider: "ollama" or "openai"          |
| `--model`      | Override model name (for either provider)       |
| `--server`     | Override Ollama server address                  |
| `--api-key`    | Override OpenAI API key                         |
| `--delay`      | Add delay between files in seconds (helps with API rate limits) |
| `--log-level`  | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Example with Multiple Exclude Folders

```bash
./generate_tags.py --exclude "/path/to/exclude1" --exclude "/path/to/exclude2"
```

### Example Using OpenAI with Delay

When using OpenAI, you might hit rate limits. Adding a delay between files helps:

```bash
./generate_tags.py --provider openai --api-key "your-api-key" --delay 5
```

This adds a 5-second pause between processing each file, which reduces the chances of hitting rate limits.

## LLM Provider Support

The script supports two LLM providers for generating tags:

1. **Ollama** (default): Use a local Ollama server running on your machine or network
2. **OpenAI**: Use OpenAI's API (requires an API key)

### Configuring OpenAI

To use OpenAI instead of Ollama, you need to:

1. Install the OpenAI Python package:
   ```bash
   pip install openai
   ```

2. Set your OpenAI API key in the config.yaml file:
   ```yaml
   LLM_PROVIDER: "openai"
   OPENAI_API_KEY: "your-api-key-here"
   OPENAI_MODEL: "gpt-3.5-turbo"  # default, or use "gpt-4" for better results
   ```

3. Alternatively, you can set these via command line:
   ```bash
   ./generate_tags.py --provider openai --api-key "your-api-key-here" --model "gpt-4"
   ```

The script uses "gpt-3.5-turbo" as the default OpenAI model, which provides a good balance between availability, cost, and quality. For even better tagging quality, you can use "gpt-4" or "o1-mini", though these may have different quota limits or costs.

## Logging

The script automatically creates log files in a `logs` directory next to the script. Each log file is named with the current date (`generate_tags_YYYY-MM-DD.log`) and contains detailed information about the script's execution, including any errors or warnings.

You can control the logging verbosity with these command-line options:
- `--debug`: Enable detailed debug logging
- `--log-level`: Set a specific logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Example:
```bash
./generate_tags.py --log-level WARNING  # Only show warnings and errors
```

Log files are helpful for troubleshooting and reviewing what happened during previous runs.

## Scheduling

### On Unix/Linux/macOS (cron)

To run the script daily at 1 AM:

```
0 1 * * * /path/to/generate_tags.py
```

### On Windows (Task Scheduler)

1. Open Task Scheduler
2. Create a new Basic Task
3. Set the trigger to "Daily"
4. Set the action to "Start a program"
5. Program: `python`
6. Arguments: `C:\path\to\generate_tags.py`

## License

MIT License
