import subprocess

# Scripts run in order included below
scripts = [
    "src/generate_artifact_statistics.py",
    "src/train_electra.py",
    "src/train_mce.py",
    "src/evaluate_electra.py",
    "src/evaluate_mce.py",
]

def run_script(script_path):
    """
    Run a Python script and print its output in real-time.
    """
    print(f"Running {script_path}...")
    try:
        result = subprocess.run(["python", script_path], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"Finished {script_path}\n{'='*50}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_path}:\n{e.stderr}")
        exit(1)

def main():
    """
    Run all scripts in the specified order.
    """
    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()
