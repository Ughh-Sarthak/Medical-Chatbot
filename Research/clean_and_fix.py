import json
import os
import sys

# Define the path
notebook_path = r"S:/MedicalChatbot/Research/trials.ipynb"

# Check existence
if not os.path.exists(notebook_path):
    # Try with lowercase drive letter just in case 
    notebook_path = r"s:/MedicalChatbot/Research/trials.ipynb"
    if not os.path.exists(notebook_path):
        print(f"Error: File not found at {notebook_path}")
        # Try to list directory to see what's there
        dir_path = os.path.dirname(notebook_path)
        if os.path.exists(dir_path):
            print(f"Directory {dir_path} exists. Contents:")
            print(os.listdir(dir_path))
        else:
            print(f"Directory {dir_path} does not exist.")
        sys.exit(1)

print(f"Found file at {notebook_path}")

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("Loaded notebook JSON.")

    modified = False

    # 1. Clean huge output
    if 'cells' in nb:
        for cell in nb['cells']:
            # Check for loader.load()
            if cell.get('cell_type') == 'code':
                source_text = "".join(cell.get('source', []))
                if "loader.load()" in source_text:
                    print("Found 'loader.load()' cell. Clearing outputs.")
                    cell['outputs'] = []
                    modified = True
                
                # 2. Fix prompt variable
                if "ChatPromptTemplate.from_messages" in source_text:
                    print("Found ChatPromptTemplate cell.")
                    new_source = []
                    cell_modified = False
                    for line in cell.get('source', []):
                        if '("human", "{question}")' in line:
                            print(f"Found incorrect line: {line.strip()}")
                            new_line = line.replace('{question}', '{input}')
                            print(f"Replaced with: {new_line.strip()}")
                            new_source.append(new_line)
                            cell_modified = True
                        elif '("human", "{question}"),' in line: # Handle potential comma
                             print(f"Found incorrect line: {line.strip()}")
                             new_line = line.replace('{question}', '{input}')
                             print(f"Replaced with: {new_line.strip()}")
                             new_source.append(new_line)
                             cell_modified = True
                        else:
                            new_source.append(line)
                    
                    if cell_modified:
                        cell['source'] = new_source
                        modified = True
                        print("Applied fix to prompt variable.")

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Draft saved successfully.")
    else:
        print("No changes needed.")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
