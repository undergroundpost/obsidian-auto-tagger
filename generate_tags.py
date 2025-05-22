#!/usr/bin/env python3
# Generate tags for Obsidian notes using AI (Ollama or OpenAI)
# Scans for notes created/modified on the previous day and adds relevant tags

import os, sys, re, json, logging, requests, time, yaml
from datetime import datetime, timedelta
from pathlib import Path
from dateutil import parser

# Configure logger - will be properly initialized in __main__
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file with defaults"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default configuration
    default_config = {
        "INPUT_FOLDER": os.path.expanduser("~/Documents/Notes"),
        "EXCLUDE_FOLDERS": [os.path.expanduser("~/Documents/Notes/AI")],
        "LLM_PROVIDER": "ollama", 
        "OLLAMA_MODEL": "gemma3:12b",
        "OLLAMA_SERVER_ADDRESS": "http://localhost:11434",
        "OLLAMA_CONTEXT_WINDOW": 32000,
        "OPENAI_API_KEY": "",
        "OPENAI_MODEL": "gpt-3.5-turbo",
        "OPENAI_MAX_TOKENS": 4000
    }
    
    # Check config file locations in priority order
    for config_path in [
        os.path.join(script_dir, "config.yaml"),
        os.path.expanduser("~/.config/generate_tags/config.yaml"),
        "/etc/generate_tags/config.yaml"
    ]:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                    logger.info(f"Loaded configuration from {config_path}")
                    config = {**default_config, **user_config}
                    
                    # Handle backward compatibility
                    if "EXCLUDE_FOLDER" in user_config and "EXCLUDE_FOLDERS" not in user_config:
                        config["EXCLUDE_FOLDERS"] = [user_config["EXCLUDE_FOLDER"]]
                    
                    return config
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
    
    logger.info("No config file found, using default configuration")
    return default_config

def get_file_creation_time(file_path):
    """Get file creation time with platform-specific handling"""
    if sys.platform == 'darwin':  # macOS
        try:
            import subprocess
            result = subprocess.run(['stat', '-f', '%B', file_path], capture_output=True, text=True, check=True)
            return datetime.fromtimestamp(float(result.stdout.strip()))
        except Exception as e:
            logger.warning(f"Error getting macOS file creation time: {e}")
            return datetime.fromtimestamp(os.path.getctime(file_path))
    else:  # Windows/Linux
        return datetime.fromtimestamp(os.path.getctime(file_path))

def find_files_to_process(input_folder, exclude_folders):
    """Find markdown files that need processing based on processed timestamp"""
    md_files = []
    skipped_files = 0
    already_processed_files = 0
    
    logger.info(f"Searching for Markdown files to process in: {input_folder}")
    logger.info(f"Excluding folders: {exclude_folders}")
    
    # Time thresholds
    now = datetime.now()
    recent_threshold = now - timedelta(minutes=15)  # For ignore buffer
    
    for root, _, files in os.walk(input_folder):
        # Skip excluded folders
        if any(root.startswith(exclude_folder) for exclude_folder in exclude_folders):
            continue
            
        for file in files:
            if not file.endswith('.md'):
                continue
                
            file_path = os.path.join(root, file)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                frontmatter, _ = parse_frontmatter(content)
                
                # Case 1: File has never been processed
                if 'processed' not in frontmatter:
                    logger.debug(f"Including unprocessed file: {file_path}")
                    md_files.append(file_path)
                    continue
                
                # Case 2: File has been processed before
                try:
                    processed_time = parser.parse(str(frontmatter['processed']))
                    
                    # Apply ignore buffer - always process very recently processed files
                    if processed_time > recent_threshold:
                        logger.debug(f"Including recently processed file (ignore buffer): {file_path}")
                        md_files.append(file_path)
                        continue
                    
                    # Check if modified since processed timestamp (plus cooldown)
                    cooldown_threshold = processed_time + timedelta(minutes=15)
                    
                    if file_mtime > cooldown_threshold:
                        logger.debug(f"Including file modified after cooldown: {file_path}")
                        md_files.append(file_path)
                    else:
                        logger.debug(f"Skipping already processed file: {file_path}")
                        already_processed_files += 1
                        
                except (ValueError, TypeError):
                    # Can't parse timestamp - include it to be safe
                    logger.debug(f"Including file with invalid processed timestamp: {file_path}")
                    md_files.append(file_path)
                    
            except Exception as e:
                # Error reading file - include it to be safe
                logger.debug(f"Error checking frontmatter: {e}")
                md_files.append(file_path)
    
    # Log results
    if not md_files:
        logger.warning("No files were found for processing!")
        if already_processed_files > 0:
            logger.warning(f"{already_processed_files} files were skipped (already processed)")
    else:
        logger.info(f"Found {len(md_files)} files for processing")
        for file_path in md_files[:5]:  # Show first 5 files
            logger.info(f"  - {os.path.basename(file_path)}")
        if len(md_files) > 5:
            logger.info(f"  - ... and {len(md_files) - 5} more files")
            
        if already_processed_files > 0:
            logger.info(f"Additionally, {already_processed_files} files were skipped (already processed)")
    
    return md_files

def clean_note_content(content):
    """Remove dataview blocks and template syntax from note content"""
    # Remove dataview blocks
    content = re.sub(r'```dataview(?:js)?\n.*?```', '', content, flags=re.DOTALL)
    
    # Remove templating code
    content = re.sub(r'<%.*?%>', '', content, flags=re.DOTALL)
    content = re.sub(r'<<.*?>>', '', content, flags=re.DOTALL)
    content = re.sub(r'\{\{.*?\}\}', '', content, flags=re.DOTALL)
    
    return content

def parse_frontmatter(content):
    """Parse YAML frontmatter from markdown content"""
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    
    if frontmatter_match:
        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
            rest_content = content[frontmatter_match.end():]
            return frontmatter, rest_content
        except yaml.YAMLError:
            logger.warning("Invalid YAML frontmatter")
    
    return {}, content

def update_frontmatter_with_tags(content, tags, mark_as_processed=True):
    """Update note frontmatter with new tags (avoiding duplicates)"""
    frontmatter, rest_content = parse_frontmatter(content)
    
    # Parse tag string into list
    tag_list = [tag.strip('#') for tag in tags.split() if tag.startswith('#')]
    
    # Only proceed if we have tags to add or if we're explicitly marking as processed
    if not tag_list and not mark_as_processed:
        return content
    
    # Update or create tags in frontmatter
    if tag_list:  # Only update tags if we have new tags
        if 'tags' in frontmatter:
            existing_tags = frontmatter['tags']
            
            # Normalize tags format
            if existing_tags is None:
                existing_tags = []
            elif isinstance(existing_tags, str):
                existing_tags = [existing_tags]
            elif not isinstance(existing_tags, list):
                existing_tags = [str(existing_tags)]
                
            # Add new tags that don't already exist (case-insensitive)
            existing_lower = [t.lower() for t in existing_tags]
            for tag in tag_list:
                if tag.lower() not in existing_lower:
                    existing_tags.append(tag)
                    logger.debug(f"Adding tag: {tag}")
                    
            frontmatter['tags'] = existing_tags
        else:
            frontmatter['tags'] = tag_list
    
    # Add or update processed timestamp only if explicitly requested
    if mark_as_processed:
        frontmatter['processed'] = datetime.now().isoformat()
    
    # Generate new frontmatter with clean formatting
    new_frontmatter = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
    
    # Clean up extra newlines
    if rest_content.strip():
        rest_content = rest_content.lstrip()
        updated_content = f"---\n{new_frontmatter}---\n\n{rest_content}"
    else:
        updated_content = f"---\n{new_frontmatter}---\n"
    
    return updated_content

def collect_all_vault_tags(input_folder, exclude_folders):
    """Collect all unique tags from the vault for better tag consistency"""
    logger.info("Collecting all tags from the vault...")
    all_tags = set()
    files_processed = 0
    
    for root, _, files in os.walk(input_folder):
        if any(root.startswith(exclude_folder) for exclude_folder in exclude_folders):
            continue
            
        for file in files:
            if not file.endswith('.md'):
                continue
                
            files_processed += 1
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get tags from frontmatter
                frontmatter, _ = parse_frontmatter(content)
                if 'tags' in frontmatter and frontmatter['tags']:
                    tags = frontmatter['tags']
                    
                    if isinstance(tags, str):
                        all_tags.add(tags.lower())
                    elif isinstance(tags, list):
                        all_tags.update(t.lower() for t in tags if t)
                    else:
                        all_tags.add(str(tags).lower())
                
                # Get inline tags
                all_tags.update(tag.lower() for tag in re.findall(r'(?<!\S)#([a-zA-Z0-9_]+)', content))
                        
            except Exception as e:
                logger.debug(f"Error processing {file}: {e}")
    
    logger.info(f"Processed {files_processed} files, found {len(all_tags)} unique tags")
    return sorted(all_tags)

def generate_tags_with_ai(content, prompt, vault_tags, config, provider=None):
    """Generate tags using either Ollama or OpenAI"""
    provider = provider or config.get("LLM_PROVIDER", "ollama").lower()
    
    # Format existing/vault tags for the prompt
    existing_tags = []
    frontmatter, _ = parse_frontmatter(content)
    if 'tags' in frontmatter and frontmatter['tags']:
        tags = frontmatter['tags']
        if isinstance(tags, list):
            existing_tags = tags
        elif isinstance(tags, str):
            existing_tags = [tags]
        else:
            existing_tags = [str(tags)]
    
    if vault_tags:
        tag_list = vault_tags
    else:
        tag_list = existing_tags
        
    tags_formatted = ' '.join(['#' + tag for tag in tag_list]) if tag_list else "No preexisting tags"
    
    # Update prompt with tag list
    updated_prompt = prompt.replace(
        "# PREEXISTING TAGS\n[list of preexisting tags will be provided here]", 
        f"# PREEXISTING TAGS\n{tags_formatted}"
    )
    
    # Call appropriate provider API
    if provider == "openai":
        logger.info(f"Using OpenAI ({config.get('OPENAI_MODEL')}) for tag generation")
        
        try:
            import openai
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            return ""
        
        api_key = config.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("No OpenAI API key provided")
            return ""
            
        model = config.get("OPENAI_MODEL", "gpt-3.5-turbo")
        max_tokens = config.get("OPENAI_MAX_TOKENS", 4000)
        client = openai.OpenAI(api_key=api_key)
        
        # Setup message payload
        messages = [
            {"role": "system", "content": updated_prompt},
            {"role": "user", "content": content}
        ]
        
        # Retry mechanism for rate limits
        max_retries = 5
        base_delay = 20.0
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                    
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    logger.info(f"Rate limit exceeded, retrying in {base_delay} seconds...")
                    time.sleep(base_delay)
                else:
                    logger.error(f"Maximum retries reached for rate limits")
                    raise e
                    
        logger.error("Failed to get a response from OpenAI")
        return ""
        
    else:  # Default to Ollama
        logger.info(f"Using Ollama ({config.get('OLLAMA_MODEL')}) for tag generation")
        
        model = config.get("OLLAMA_MODEL", "gemma3:12b")
        server_address = config.get("OLLAMA_SERVER_ADDRESS", "http://localhost:11434")
        context_window = config.get("OLLAMA_CONTEXT_WINDOW", 32000)
        
        # Build the API payload
        payload = {
            "model": model,
            "prompt": updated_prompt + "\n\n" + content,
            "stream": False,
            "options": {
                "num_ctx": context_window,
                "cache_prompt": False
            }
        }
        
        try:
            response = requests.post(
                f"{server_address}/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=900
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    return result['response'].strip()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            
        return ""

def main():
    """Main function to process markdown files and update tags"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate and update tags for Obsidian notes')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging')
    parser.add_argument('--input', help='Override input folder')
    parser.add_argument('--exclude', action='append', help='Override exclude folders (can be used multiple times)')
    parser.add_argument('--model', help='Override model name')
    parser.add_argument('--server', help='Override Ollama server address')
    parser.add_argument('--provider', choices=['ollama', 'openai'], help='LLM provider (ollama or openai)')
    parser.add_argument('--api-key', help='Override OpenAI API key')
    parser.add_argument('--delay', type=float, default=0, help='Delay between processing files (seconds)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                      default='INFO', help='Set logging level')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of files to process (0 for no limit)')
    parser.add_argument('--batch-mode', action='store_true', 
                      help='Enable batch mode for processing large numbers of files')
    parser.add_argument('--batch-size', type=int, default=20, 
                      help='Number of files to process in each batch (with batch mode)')
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("=== Starting generate_tags.py ===")
    
    # Load and override configuration
    config = load_config()
    if args.input: config["INPUT_FOLDER"] = args.input
    if args.exclude: config["EXCLUDE_FOLDERS"] = args.exclude
    if args.provider: config["LLM_PROVIDER"] = args.provider
    if args.api_key: config["OPENAI_API_KEY"] = args.api_key
    if args.model:
        if config.get("LLM_PROVIDER") == "openai" or args.provider == "openai":
            config["OPENAI_MODEL"] = args.model
        else:
            config["OLLAMA_MODEL"] = args.model
    if args.server: config["OLLAMA_SERVER_ADDRESS"] = args.server
    
    # Extract common config values
    input_folder = config["INPUT_FOLDER"]
    exclude_folders = config["EXCLUDE_FOLDERS"]
    llm_provider = config["LLM_PROVIDER"]
    
    # Get prompt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_path = os.path.join(script_dir, "generate_tags.md")
    
    # Log basic configuration
    logger.info(f"Configuration: Input={input_folder}, Provider={llm_provider}")
    if llm_provider.lower() == "openai":
        if args.delay > 0:
            logger.info(f"Using OpenAI with {args.delay}s delay between files")
        else:
            logger.info("Using OpenAI with no delay (add --delay 5 to avoid rate limits)")
    
    # Verify prompt file exists
    if not os.path.exists(prompt_file_path):
        logger.error(f"Prompt file not found: {prompt_file_path}")
        return
    
    # Load prompt file
    try:
        with open(prompt_file_path, 'r') as f:
            tag_prompt = f.read()
    except Exception as e:
        logger.error(f"Error reading prompt file: {e}")
        return
    
    # Find files to process using new method
    try:
        md_files = find_files_to_process(input_folder, exclude_folders)
    except Exception as e:
        logger.error(f"Error finding files: {e}")
        return
    
    if not md_files:
        logger.info("No files found for processing. Exiting.")
        return
    
    # Apply file limit if specified
    if args.limit > 0 and len(md_files) > args.limit:
        logger.info(f"Limiting processing to {args.limit} files (out of {len(md_files)} found)")
        md_files = md_files[:args.limit]
        
    # Collect vault tags
    try:
        vault_tags = collect_all_vault_tags(input_folder, exclude_folders)
    except Exception as e:
        logger.error(f"Error collecting vault tags: {e}")
        vault_tags = []
    
    # Process each file - with batch mode support
    tags_added = 0
    files_with_errors = 0
    total_files = len(md_files)

    try:
        # Check if batch mode is enabled and needed
        if args.batch_mode and len(md_files) > args.batch_size:
            logger.info(f"Batch mode: Processing {len(md_files)} files in batches of {args.batch_size}")
            
            total_batches = (len(md_files) + args.batch_size - 1) // args.batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(md_files))
                batch_files = md_files[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num+1}/{total_batches} ({len(batch_files)} files)")
                
                # Process each file in this batch
                for index, file_path in enumerate(batch_files):
                    batch_index = start_idx + index
                    filename = os.path.basename(file_path)
                    progress = f"[{batch_index+1}/{total_files}]"
                    logger.info(f"{progress} Processing: {filename}")
                    
                    try:
                        # Read and clean content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        clean_content = clean_note_content(content)
                        
                        # Generate tags
                        generated_tags = generate_tags_with_ai(clean_content, tag_prompt, vault_tags, config)
                        
                        if not generated_tags:
                            logger.warning(f"{progress} No tags generated for {filename} - not marking as processed")
                            continue
                            
                        logger.info(f"{progress} Generated tags: {generated_tags}")
                        
                        # Update frontmatter with tags AND mark as processed
                        updated_content = update_frontmatter_with_tags(content, generated_tags, mark_as_processed=True)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(updated_content)
                            
                        logger.info(f"{progress} Updated tags in {filename}")
                        tags_added += 1
                        
                        # Add delay between files if specified
                        if args.delay > 0 and llm_provider.lower() == "openai" and index < len(batch_files) - 1:
                            logger.info(f"Waiting {args.delay}s before next file...")
                            time.sleep(args.delay)
                            
                    except Exception as e:
                        logger.error(f"{progress} Error processing {filename}: {e} - not marking as processed")
                        files_with_errors += 1
                
                # Pause between batches
                if batch_num < total_batches - 1:
                    logger.info(f"Batch {batch_num+1} complete. Waiting 30 seconds before next batch...")
                    time.sleep(30)  # Pause between batches to avoid overloading the LLM service
        
        else:
            # Standard processing (no batching)
            for index, file_path in enumerate(md_files):
                filename = os.path.basename(file_path)
                progress = f"[{index+1}/{total_files}]"
                logger.info(f"{progress} Processing: {filename}")
                
                try:
                    # Read and clean content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    clean_content = clean_note_content(content)
                    
                    # Generate tags
                    generated_tags = generate_tags_with_ai(clean_content, tag_prompt, vault_tags, config)
                    
                    if not generated_tags:
                        logger.warning(f"{progress} No tags generated for {filename} - not marking as processed")
                        continue
                        
                    logger.info(f"{progress} Generated tags: {generated_tags}")
                    
                    # Update frontmatter with tags AND mark as processed
                    updated_content = update_frontmatter_with_tags(content, generated_tags, mark_as_processed=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                        
                    logger.info(f"{progress} Updated tags in {filename}")
                    tags_added += 1
                    
                    # Add delay between files if specified
                    if args.delay > 0 and llm_provider.lower() == "openai" and index < total_files - 1:
                        logger.info(f"Waiting {args.delay}s before next file...")
                        time.sleep(args.delay)
                        
                except Exception as e:
                    logger.error(f"{progress} Error processing {filename}: {e} - not marking as processed")
                    files_with_errors += 1
                    
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")

    # Log summary
    logger.info(f"=== Summary: {tags_added}/{total_files} files tagged, {files_with_errors} errors ===")
    if llm_provider.lower() == "openai" and args.delay == 0 and files_with_errors > 0:
        logger.info("Tip: Use --delay 5 to avoid OpenAI API rate limits")
    if tags_added > 10 and not args.batch_mode:
        logger.info("Tip: Use --batch-mode for processing many files")

if __name__ == "__main__":
    # Set up logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"generate_tags_{datetime.now().strftime('%Y-%m-%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)
