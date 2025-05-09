#!/usr/bin/env python3
# ==============================================================================
# Script: generate_tags.py
#
# Description:
#   Scans an input folder for .md files that were created or modified on the previous
#   calendar day. For each file found, it sends the file's text along with
#   a tagging system prompt to your Ollama server or OpenAI. The generated tags are
#   then appended to the note's frontmatter (YAML properties).
#
# Requirements:
#   - requests (for making API calls)
#   - python-dateutil (for date handling)
#   - pyyaml (for frontmatter parsing)
#   - openai (optional, for OpenAI API access)
#
# Configuration:
#   This script uses a config.yaml file for settings. Create this file in the same
#   directory as the script, or use command-line parameters to override defaults.
# ==============================================================================

import os
import sys
import re
import json
import logging
import requests
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import time
from dateutil import parser

# Configure logging - will be properly initialized in __main__
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file or use defaults."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default configuration
    default_config = {
        "INPUT_FOLDER": os.path.expanduser("~/Documents/Notes"),
        "EXCLUDE_FOLDERS": [os.path.expanduser("~/Documents/Notes/AI")],
        "LLM_PROVIDER": "ollama",  # Options: "ollama", "openai"
        "OLLAMA_MODEL": "gemma3:12b",
        "OLLAMA_SERVER_ADDRESS": "http://localhost:11434",
        "OLLAMA_CONTEXT_WINDOW": 32000,
        "OPENAI_API_KEY": "",  # User must provide their OpenAI API key
        "OPENAI_MODEL": "gpt-3.5-turbo",  # Default to the widely available gpt-3.5-turbo model
        "OPENAI_MAX_TOKENS": 4000
    }
    
    # Possible config file locations (in order of preference)
    config_paths = [
        os.path.join(script_dir, "config.yaml"),                     # Same directory as script
        os.path.expanduser("~/.config/generate_tags/config.yaml"),   # User config directory
        "/etc/generate_tags/config.yaml"                             # System-wide config
    ]
    
    # Try to load from config file
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        logger.info(f"Loaded configuration from {config_path}")
                        # Merge user config with defaults (user config takes precedence)
                        config = {**default_config, **user_config}
                        
                        # Handle backward compatibility for single EXCLUDE_FOLDER
                        if "EXCLUDE_FOLDER" in user_config and "EXCLUDE_FOLDERS" not in user_config:
                            config["EXCLUDE_FOLDERS"] = [user_config["EXCLUDE_FOLDER"]]
                            logger.info("Converting EXCLUDE_FOLDER to EXCLUDE_FOLDERS list")
                        
                        return config
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
    
    # If no config file found, use defaults
    logger.info("No config file found, using default configuration")
    logger.info("To customize, create a config.yaml file in the script directory")
    return default_config

def get_previous_day_boundaries(override_date=None):
    """Get the time boundaries for the previous calendar day, or a specific day if provided."""
    if override_date:
        try:
            target_date = parser.parse(override_date).date()
            logger.info(f"Using override date: {target_date}")
        except ValueError:
            logger.error(f"Invalid override date format: {override_date}. Using previous day instead.")
            today = datetime.now()
            target_date = (today - timedelta(days=1)).date()
    else:
        today = datetime.now()
        target_date = (today - timedelta(days=1)).date()
    
    start_boundary = datetime.combine(target_date, datetime.min.time())
    end_boundary = datetime.combine(target_date, datetime.max.time())
    
    logger.info(f"Target date: {target_date}")
    logger.info(f"Start boundary: {start_boundary}")
    logger.info(f"End boundary: {end_boundary}")
    
    return start_boundary, end_boundary

def get_file_creation_time(file_path):
    """Get the actual file creation time, with special handling for macOS."""
    # Check if we're on macOS
    if sys.platform == 'darwin':
        try:
            # Use macOS-specific stat command to get creation time
            import subprocess
            result = subprocess.run(['stat', '-f', '%B', file_path], 
                                   capture_output=True, text=True, check=True)
            creation_timestamp = float(result.stdout.strip())
            return datetime.fromtimestamp(creation_timestamp)
        except (subprocess.SubprocessError, ValueError) as e:
            logger.warning(f"Error getting macOS file creation time: {e}")
            # Fall back to getctime if the subprocess approach fails
            return datetime.fromtimestamp(os.path.getctime(file_path))
    else:
        # For other platforms, use standard getctime (which works correctly on Windows)
        return datetime.fromtimestamp(os.path.getctime(file_path))

def find_md_files_from_previous_day(input_folder, exclude_folders, start_boundary, end_boundary):
    """Find all .md files created OR modified on the previous day."""
    md_files = []
    
    logger.info(f"Searching for Markdown files in: {input_folder}")
    logger.info(f"Using date range: {start_boundary.date()} to {end_boundary.date()}")
    logger.info(f"Excluding folders: {exclude_folders}")
    
    for root, _, files in os.walk(input_folder):
        # Skip excluded folders
        skip_folder = False
        for exclude_folder in exclude_folders:
            if root.startswith(exclude_folder):
                skip_folder = True
                break
                
        if skip_folder:
            continue
            
        for file in files:
            if not file.endswith('.md'):
                continue
                
            file_path = os.path.join(root, file)
            
            # Get proper creation time (platform-specific)
            file_ctime = get_file_creation_time(file_path)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Log file timestamps for debugging
            logger.debug(f"File: {file_path}")
            logger.debug(f"  Creation time: {file_ctime}")
            logger.debug(f"  Modified time: {file_mtime}")
            
            # Check if system timestamps are within range
            if (start_boundary <= file_ctime <= end_boundary) or (start_boundary <= file_mtime <= end_boundary):
                logger.debug(f"  MATCH: File timestamp within range")
                md_files.append(file_path)
                continue
            
            # If not caught by filesystem timestamps, check frontmatter
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                frontmatter, _ = parse_frontmatter(content)
                
                # Check for creation date in frontmatter (various possible fields)
                for date_field in ['created', 'date', 'creation_date', 'createdAt']:
                    if date_field in frontmatter and frontmatter[date_field]:
                        try:
                            # Parse the date string in frontmatter (various formats)
                            fm_date = parser.parse(str(frontmatter[date_field]))
                            
                            # Convert to datetime if it's a date object
                            if not isinstance(fm_date, datetime):
                                fm_date = datetime.combine(fm_date, datetime.min.time())
                            
                            logger.debug(f"  Frontmatter {date_field}: {fm_date.date()}")
                                
                            # Check if frontmatter date is within range
                            if start_boundary.date() <= fm_date.date() <= end_boundary.date():
                                logger.debug(f"  MATCH: Frontmatter date within range")
                                md_files.append(file_path)
                                break
                        except (ValueError, TypeError) as e:
                            logger.debug(f"  Error parsing {date_field}: {e}")
                            # Skip unparseable dates
                            pass
                            
            except Exception as e:
                logger.debug(f"Error checking frontmatter dates for {file}: {e}")
    
    if not md_files:
        logger.warning("No files were found matching the date criteria!")
    else:
        logger.info("Files found:")
        for file_path in md_files:
            logger.info(f"  - {os.path.basename(file_path)}")
    
    logger.info(f"Found {len(md_files)} Markdown files created or modified on the previous day")
    return md_files

def clean_note_content(content):
    """Clean note content by removing dataview blocks and template syntax."""
    # Remove dataview or dataviewjs code blocks
    content = re.sub(r'```dataview(?:js)?\n.*?```', '', content, flags=re.DOTALL)
    
    # Remove inline Obsidian templating code
    content = re.sub(r'<%.*?%>', '', content, flags=re.DOTALL)
    content = re.sub(r'<<.*?>>', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)
    
    return content

def parse_frontmatter(content):
    """Parse YAML frontmatter from markdown content."""
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    
    if frontmatter_match:
        frontmatter_content = frontmatter_match.group(1)
        rest_content = content[frontmatter_match.end():]
        try:
            frontmatter = yaml.safe_load(frontmatter_content) or {}
            return frontmatter, rest_content
        except yaml.YAMLError:
            logger.warning("Invalid YAML frontmatter, treating as if no frontmatter exists")
    
    # No frontmatter or invalid YAML
    return {}, content

def update_frontmatter_with_tags(content, tags):
    """Update note frontmatter with the provided tags, avoiding any duplication."""
    frontmatter, rest_content = parse_frontmatter(content)
    
    # Parse tag string into list (strip the '#' prefix)
    tag_list = [tag.strip('#') for tag in tags.split() if tag.startswith('#')]
    
    # Update or create tags in frontmatter
    if 'tags' in frontmatter:
        # If frontmatter already has tags
        existing_tags = frontmatter['tags']
        
        # Handle different formats of existing tags
        if existing_tags is None:
            existing_tags = []
        elif isinstance(existing_tags, str):
            existing_tags = [existing_tags]  # Convert single string tag to list
        elif not isinstance(existing_tags, list):
            # Handle case where tags might be another type
            logger.warning(f"Unexpected tags format: {type(existing_tags)}, converting to list")
            existing_tags = [str(existing_tags)]
            
        # Add new tags that don't already exist (case-insensitive comparison)
        existing_lower = [t.lower() for t in existing_tags]
        for tag in tag_list:
            if tag.lower() not in existing_lower:
                existing_tags.append(tag)
                logger.debug(f"Adding new tag: {tag}")
            else:
                logger.debug(f"Tag already exists (skipping): {tag}")
                
        frontmatter['tags'] = existing_tags
    else:
        # If no tags exist yet
        frontmatter['tags'] = tag_list
    
    # Generate new frontmatter
    new_frontmatter = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
    
    # Clean up extra newlines - only keep one newline after the frontmatter
    # If there's content in the rest_content
    if rest_content.strip():
        # Remove leading whitespace/newlines from the content
        rest_content = rest_content.lstrip()
        # Ensure exactly one newline between frontmatter and content
        updated_content = f"---\n{new_frontmatter}---\n\n{rest_content}"
    else:
        # If there's no content, don't add extra newlines
        updated_content = f"---\n{new_frontmatter}---\n"
    
    return updated_content

def collect_all_vault_tags(input_folder, exclude_folders):
    """Collect all unique tags from all markdown files in the Obsidian vault."""
    logger.info("Collecting all tags from the vault...")
    all_tags = set()  # Use a set to store unique tags
    files_processed = 0
    
    for root, _, files in os.walk(input_folder):
        # Skip excluded folders
        skip_folder = False
        for exclude_folder in exclude_folders:
            if root.startswith(exclude_folder):
                skip_folder = True
                break
                
        if skip_folder:
            continue
            
        for file in files:
            if not file.endswith('.md'):
                continue
                
            file_path = os.path.join(root, file)
            files_processed += 1
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse frontmatter to extract tags
                frontmatter, _ = parse_frontmatter(content)
                
                if 'tags' in frontmatter and frontmatter['tags']:
                    tags = frontmatter['tags']
                    
                    # Handle different tag formats
                    if isinstance(tags, str):
                        tags = [tags]
                    elif isinstance(tags, list):
                        pass  # Already in the right format
                    else:
                        try:
                            tags = [str(tags)]
                        except:
                            tags = []
                    
                    # Add tags to the set
                    for tag in tags:
                        if tag:  # Skip empty tags
                            all_tags.add(tag.lower())  # Normalize to lowercase
                
                # Also look for inline tags like #tag in the note content
                inline_tags = re.findall(r'(?<!\S)#([a-zA-Z0-9_]+)', content)
                for tag in inline_tags:
                    if tag:  # Skip empty tags
                        all_tags.add(tag.lower())  # Normalize to lowercase
                        
            except Exception as e:
                logger.debug(f"Error processing {file}: {e}")
    
    logger.info(f"Processed {files_processed} files, found {len(all_tags)} unique tags")
    return sorted(all_tags)  # Return as sorted list

def call_ollama(text, prompt, vault_tags=None, model=None, server_address=None, context_window=None):
    """Call the Ollama API with the given text and prompt."""
    # Use provided parameters or fall back to global values (for backward compatibility)
    logger.info(f"Calling Ollama API with model: {model}")
    
    # Extract existing tags from content for the prompt
    frontmatter, _ = parse_frontmatter(text)
    existing_tags = []
    
    if 'tags' in frontmatter:
        tags = frontmatter['tags']
        if tags:
            if isinstance(tags, list):
                existing_tags = tags
            elif isinstance(tags, str):
                existing_tags = [tags]
            else:
                try:
                    existing_tags = [str(tags)]
                except:
                    logger.warning("Could not convert tags to list, using empty list")
    
    # Format the tags for the prompt
    # If vault_tags is provided, use it; otherwise just use existing tags from the current note
    if vault_tags:
        preexisting_tags = vault_tags
        # Format as space-separated hashtags
        preexisting_tags_formatted = ' '.join(['#' + tag for tag in preexisting_tags])
    else:
        preexisting_tags_formatted = ' '.join(['#' + tag for tag in existing_tags]) if existing_tags else "No preexisting tags"
    
    # Update the prompt with the list of preexisting tags
    updated_prompt = prompt.replace("# PREEXISTING TAGS\n[list of preexisting tags will be provided here]", 
                                   f"# PREEXISTING TAGS\n{preexisting_tags_formatted}")
    
    # Build the JSON payload
    payload = {
        "model": model,
        "prompt": updated_prompt + "\n\n" + text,
        "stream": False,
        "options": {
            "num_ctx": context_window,
            "cache_prompt": False
        }
    }
    
    try:
        logger.info(f"Sending request to: {server_address}/api/generate")
        response = requests.post(
            f"{server_address}/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                return result['response'].strip()
            else:
                logger.error("API response did not contain 'response' field")
                return ""
        else:
            logger.error(f"API call failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return ""
    except Exception as e:
        logger.error(f"Error calling Ollama API: {e}")
        return ""

def call_openai(text, prompt, vault_tags=None, model=None, api_key=None, max_tokens=None):
    """Call the OpenAI API with the given text and prompt."""
    try:
        import openai
    except ImportError:
        logger.error("OpenAI package not installed. Please run: pip install openai")
        return ""
    
    # Set API key
    if not api_key:
        logger.error("No OpenAI API key provided. Please set OPENAI_API_KEY in config.yaml or use --api-key")
        return ""
    
    # Use provided parameters or fall back to defaults
    model = model or "gpt-3.5-turbo"
    max_tokens = max_tokens or 4000
    
    logger.info(f"Calling OpenAI API with model: {model}")
    
    # Extract existing tags from content for the prompt
    frontmatter, _ = parse_frontmatter(text)
    existing_tags = []
    
    if 'tags' in frontmatter:
        tags = frontmatter['tags']
        if tags:
            if isinstance(tags, list):
                existing_tags = tags
            elif isinstance(tags, str):
                existing_tags = [tags]
            else:
                try:
                    existing_tags = [str(tags)]
                except:
                    logger.warning("Could not convert tags to list, using empty list")
    
    # Format the tags for the prompt
    # If vault_tags is provided, use it; otherwise just use existing tags from the current note
    if vault_tags:
        preexisting_tags = vault_tags
        # Format as space-separated hashtags
        preexisting_tags_formatted = ' '.join(['#' + tag for tag in preexisting_tags])
    else:
        preexisting_tags_formatted = ' '.join(['#' + tag for tag in existing_tags]) if existing_tags else "No preexisting tags"
    
    # Update the prompt with the list of preexisting tags
    updated_prompt = prompt.replace("# PREEXISTING TAGS\n[list of preexisting tags will be provided here]", 
                                   f"# PREEXISTING TAGS\n{preexisting_tags_formatted}")
    
    # Extract a preview of the text content for logging
    text_preview = text[:100] + "..." if len(text) > 100 else text
    logger.debug(f"Processing content: {text_preview}")
    
    # Set up the OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # Create the conversation with system prompt and user content
        messages = [
            {"role": "system", "content": updated_prompt},
            {"role": "user", "content": text}
        ]
        
        # Add retry mechanism for API rate limits
        max_retries = 5  # Increase from 3 to 5
        base_delay = 20.0  # seconds - much higher initial delay
        
        for attempt in range(max_retries):
            try:
                # Call the OpenAI API
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,  # Low temperature for consistent tagging
                )
                
                # Extract the generated tags
                if response.choices and len(response.choices) > 0:
                    tags_text = response.choices[0].message.content.strip()
                    logger.debug(f"OpenAI response: {tags_text}")
                    return tags_text
                else:
                    logger.error("No response content from OpenAI API")
                    return ""
                    
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    # Use a fixed high delay for rate limits
                    wait_time = base_delay
                    logger.info(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Maximum retries reached for rate limits. Consider using --provider ollama for local processing.")
                    raise e
            
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            logger.error(f"OpenAI API quota exceeded. Please check your billing details or switch to a different model.")
            logger.error(f"Consider using --provider ollama to switch to local processing.")
        elif "rate limit" in error_msg.lower():
            logger.error(f"OpenAI API rate limit exceeded. Consider adding a --delay parameter between files.")
            logger.error(f"Consider using --provider ollama for faster local processing.")
        else:
            logger.error(f"Error calling OpenAI API: {e}")
        return ""

def generate_tags(content, prompt, vault_tags, config, provider=None):
    """Generate tags using the configured LLM provider."""
    # Determine which provider to use
    provider = provider or config.get("LLM_PROVIDER", "ollama")
    
    if provider.lower() == "openai":
        logger.info("Using OpenAI for tag generation")
        return call_openai(
            content, 
            prompt, 
            vault_tags,
            config.get("OPENAI_MODEL"),
            config.get("OPENAI_API_KEY"),
            config.get("OPENAI_MAX_TOKENS")
        )
    else:  # Default to Ollama
        logger.info("Using Ollama for tag generation")
        return call_ollama(
            content, 
            prompt, 
            vault_tags,
            config.get("OLLAMA_MODEL"),
            config.get("OLLAMA_SERVER_ADDRESS"),
            config.get("OLLAMA_CONTEXT_WINDOW")
        )

def main():
    """Main function to process markdown files and update tags."""
    logger.info("=== Starting generate_tags.py ===")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate and update tags for Obsidian notes')
    parser.add_argument('--date', help='Override date to check (YYYY-MM-DD format)')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging')
    parser.add_argument('--input', help='Override input folder')
    parser.add_argument('--exclude', action='append', help='Override exclude folders (can be used multiple times)')
    parser.add_argument('--model', help='Override model name')
    parser.add_argument('--server', help='Override Ollama server address')
    parser.add_argument('--provider', choices=['ollama', 'openai'], help='Override LLM provider (ollama or openai)')
    parser.add_argument('--api-key', help='Override OpenAI API key')
    parser.add_argument('--delay', type=float, default=0, help='Add delay between processing files (seconds)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help='Set logging level')
    args = parser.parse_args()
    
    # Set log level based on args
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(getattr(logging, args.log_level))
    
    # Load configuration (with command line overrides)
    config = load_config()
    
    # Apply command line overrides
    if args.input:
        config["INPUT_FOLDER"] = args.input
    if args.exclude:
        config["EXCLUDE_FOLDERS"] = args.exclude
    if args.model:
        if config.get("LLM_PROVIDER") == "openai" or args.provider == "openai":
            config["OPENAI_MODEL"] = args.model
        else:
            config["OLLAMA_MODEL"] = args.model
    if args.server:
        config["OLLAMA_SERVER_ADDRESS"] = args.server
    if args.provider:
        config["LLM_PROVIDER"] = args.provider
    if args.api_key:
        config["OPENAI_API_KEY"] = args.api_key
    
    # Extract common config values
    INPUT_FOLDER = config["INPUT_FOLDER"]
    EXCLUDE_FOLDERS = config["EXCLUDE_FOLDERS"]
    LLM_PROVIDER = config["LLM_PROVIDER"]
    
    # Get script directory for prompt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(script_dir, "generate_tags.md")
    
    # Log configuration
    logger.info(f"Using configuration:")
    logger.info(f"  INPUT_FOLDER: {INPUT_FOLDER}")
    logger.info(f"  EXCLUDE_FOLDERS: {EXCLUDE_FOLDERS}")
    logger.info(f"  PROMPT_FILE: {prompt_file_path}")
    logger.info(f"  LLM_PROVIDER: {LLM_PROVIDER}")
    
    if LLM_PROVIDER.lower() == "ollama":
        logger.info(f"  OLLAMA_MODEL: {config.get('OLLAMA_MODEL')}")
        logger.info(f"  OLLAMA_SERVER_ADDRESS: {config.get('OLLAMA_SERVER_ADDRESS')}")
    elif LLM_PROVIDER.lower() == "openai":
        logger.info(f"  OPENAI_MODEL: {config.get('OPENAI_MODEL')}")
        if args.delay > 0:
            logger.info(f"  DELAY BETWEEN FILES: {args.delay} seconds")
        else:
            logger.info(f"  DELAY BETWEEN FILES: None (consider adding --delay 5 if you hit rate limits)")
        # Don't log the API key for security
        if not config.get("OPENAI_API_KEY"):
            logger.warning("  OPENAI_API_KEY: Not set! Tags cannot be generated without an API key.")
    
    # Check if OpenAI is requested but not installed
    if LLM_PROVIDER.lower() == "openai":
        try:
            import openai
        except ImportError:
            logger.error("OpenAI package not installed. Please run: pip install openai")
            logger.error("Exiting...")
            return
    
    # Check if the prompt file exists
    if not os.path.exists(prompt_file_path):
        logger.error(f"Prompt file not found: {prompt_file_path}")
        logger.error(f"Please place generate_tags.md in the same directory as this script")
        return
    
    # Read the prompt file
    try:
        with open(prompt_file_path, 'r') as f:
            tag_prompt = f.read()
    except Exception as e:
        logger.error(f"Error reading prompt file: {e}")
        return
    
    # Get time boundaries for the specified date or previous day
    start_boundary, end_boundary = get_previous_day_boundaries(args.date)
    
    # Find markdown files from the previous day
    try:
        md_files = find_md_files_from_previous_day(INPUT_FOLDER, EXCLUDE_FOLDERS, start_boundary, end_boundary)
    except Exception as e:
        logger.error(f"Error finding markdown files: {e}")
        return
    
    if not md_files:
        logger.info("No files found matching the date criteria. Exiting.")
        return
        
    # Collect all tags from the vault
    try:
        vault_tags = collect_all_vault_tags(INPUT_FOLDER, EXCLUDE_FOLDERS)
        logger.info(f"Collected {len(vault_tags)} unique tags from the vault")
    except Exception as e:
        logger.error(f"Error collecting vault tags: {e}")
        vault_tags = []
        logger.warning("Proceeding without vault tags")
    
    # Process each file
    tags_added = 0
    files_with_errors = 0
    total_files = len(md_files)
    
    logger.info(f"Starting to process {total_files} files")
    
    try:
        # Show a progress indicator
        for index, file_path in enumerate(md_files):
            filename = os.path.basename(file_path)
            progress = f"[{index+1}/{total_files}]"
            logger.info(f"{progress} Processing file: {filename}")
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Clean the content
                clean_content = clean_note_content(content)
                
                # Generate tags using the configured provider
                logger.info(f"{progress} Sending content to {LLM_PROVIDER} for tag generation")
                generated_tags = generate_tags(clean_content, tag_prompt, vault_tags, config)
                
                if not generated_tags:
                    logger.warning(f"{progress} No tags generated for {filename}, skipping")
                    continue
                    
                logger.info(f"{progress} Generated tags: {generated_tags}")
                
                # Update the note with new tags
                updated_content = update_frontmatter_with_tags(content, generated_tags)
                
                # Write the updated content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                    
                logger.info(f"{progress} Updated tags in {filename}")
                tags_added += 1
                
                # Add delay between files if specified (helps with API rate limits)
                if args.delay > 0 and LLM_PROVIDER.lower() == "openai" and index < total_files - 1:
                    logger.info(f"{progress} Waiting {args.delay} seconds before processing next file...")
                    time.sleep(args.delay)
                    
            except Exception as e:
                logger.error(f"{progress} Error processing {filename}: {e}")
                files_with_errors += 1
                
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        logger.info(f"Processed {index+1} of {total_files} files before interruption")
    
    # Log summary at the end
    logger.info(f"=== Processing Summary ===")
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Files successfully tagged: {tags_added}")
    logger.info(f"Files with errors: {files_with_errors}")
    
    if LLM_PROVIDER.lower() == "openai" and args.delay == 0 and files_with_errors > 0:
        logger.info(f"Tip: Consider using --delay 5 or --delay 20 to avoid OpenAI API rate limits")
        
    logger.info(f"=== Script completed successfully ===")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file name with current date
    log_file = os.path.join(logs_dir, f"generate_tags_{datetime.now().strftime('%Y-%m-%d')}.log")
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Logging to file: {log_file}")
    
    # Run the main function
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception in main function: {e}")
        sys.exit(1)
