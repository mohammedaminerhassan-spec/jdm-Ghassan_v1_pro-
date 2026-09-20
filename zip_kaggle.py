import os
import zipfile

def main():
    root_dir = r"c:\Users\Ghassan PC\Desktop\Ghassan Ai model"
    zip_path = r"c:\Users\Ghassan PC\Desktop\kaggle_project.zip"
    
    # Directories and files to explicitly exclude
    excludes = {'.git', 'convert.py', 'zip_kaggle.py', '__pycache__', '.pytest_cache'}
    
    print(f"Creating {zip_path}...")
    # Using ZIP_STORED since Parquet is already compressed and it's much faster
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for foldername, subfolders, filenames in os.walk(root_dir):
            # Skip excluded directories
            subfolders[:] = [d for d in subfolders if d not in excludes]
            
            for filename in filenames:
                if filename in excludes:
                    continue
                # Skip unnecessary logs/temp files
                if filename.endswith(".jsonl") or filename.endswith(".log"):
                    continue
                
                filepath = os.path.join(foldername, filename)
                arcname = os.path.relpath(filepath, root_dir)
                zf.write(filepath, arcname)
                
    print("Zip created successfully.")

if __name__ == "__main__":
    main()
