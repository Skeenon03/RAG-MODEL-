import os
import json

# Configuration
DOCS_PATH = "docs"
LOG_FILE = "processed_files.json"

def create_initial_log():
    processed_files = []
    
    print(f"Scanning {DOCS_PATH} to mark files as 'Done'...")
    
    if not os.path.exists(DOCS_PATH):
        print("Error: Docs folder not found.")
        return

    # Find all files currently in your docs folder
    for root, dirs, files in os.walk(DOCS_PATH):
        for file in files:
            # We must normalize the path (fix backslashes/forward slashes)
            # so it matches exactly what the new script will see.
            full_path = os.path.join(root, file)
            normalized_path = os.path.normpath(full_path)
            
            processed_files.append(normalized_path)

    # Save the log file
    with open(LOG_FILE, "w") as f:
        json.dump(processed_files, f)
        
    print(f"Success! Created '{LOG_FILE}'.")
    print(f"The new script will now ignore these {len(processed_files)} files and only add new ones.")

if __name__ == "__main__":
    create_initial_log()