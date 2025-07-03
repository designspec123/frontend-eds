import os
import sys

def merge_css_files(input_dir: str, output_file: str = "poc-ds.css"):
    with open(output_file, "w") as outfile:
        for filename in os.listdir(input_dir):
            if filename.endswith(".css"):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, "r") as infile:
                    outfile.write(f"/* ==== {filename} ==== */\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n")
    print(f"Combined CSS written to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # Use argument if provided, else default to current directory
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    merge_css_files(input_dir)
