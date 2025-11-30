# setup_venv.py
"""
Script to set up virtual environment and install dependencies.
Run: python setup_venv.py
"""
import os
import sys
import subprocess
import platform

def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result.returncode == 0

def main():
    print("="*60)
    print("Setting up Virtual Environment for Deep Learning Project")
    print("="*60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("ERROR: Python 3.8+ is required!")
        sys.exit(1)
    
    # Create virtual environment
    venv_name = "venv"
    if os.path.exists(venv_name):
        print(f"\nVirtual environment '{venv_name}' already exists.")
        response = input("Do you want to recreate it? (y/n): ")
        if response.lower() == 'y':
            print(f"Removing existing virtual environment...")
            if platform.system() == "Windows":
                run_command(f"rmdir /s /q {venv_name}", check=False)
            else:
                run_command(f"rm -rf {venv_name}", check=False)
        else:
            print("Using existing virtual environment.")
    
    if not os.path.exists(venv_name):
        print(f"\nCreating virtual environment '{venv_name}'...")
        if not run_command(f"{sys.executable} -m venv {venv_name}"):
            print("Failed to create virtual environment!")
            sys.exit(1)
    
    # Determine activation script path
    if platform.system() == "Windows":
        activate_script = os.path.join(venv_name, "Scripts", "activate")
        pip_path = os.path.join(venv_name, "Scripts", "pip")
        python_path = os.path.join(venv_name, "Scripts", "python")
    else:
        activate_script = os.path.join(venv_name, "bin", "activate")
        pip_path = os.path.join(venv_name, "bin", "pip")
        python_path = os.path.join(venv_name, "bin", "python")
    
    # Upgrade pip
    print("\nUpgrading pip...")
    run_command(f'"{python_path}" -m pip install --upgrade pip', check=False)
    
    # Install requirements
    print("\nInstalling requirements from requirements.txt...")
    if not os.path.exists("requirements.txt"):
        print("ERROR: requirements.txt not found!")
        sys.exit(1)
    
    if not run_command(f'"{pip_path}" install -r requirements.txt'):
        print("Failed to install requirements!")
        sys.exit(1)
    
    # Check for GPU support
    print("\n" + "="*60)
    print("Checking GPU Support...")
    print("="*60)
    
    # Test TensorFlow GPU
    test_script = f'''
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {{len(gpus)}} GPU(s)")
    for gpu in gpus:
        print(f"   • {{gpu}}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"     Device: {{details.get('device_name', 'Unknown')}}")
        except:
            pass
else:
    print("❌ No GPU detected. Install CUDA/cuDNN for GPU support.")
'''
    
    test_file = "test_gpu_setup.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print("\nRunning GPU test...")
    run_command(f'"{python_path}" {test_file}', check=False)
    os.remove(test_file)
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print(f"\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"  {venv_name}\\Scripts\\activate")
    else:
        print(f"  source {venv_name}/bin/activate")
    print("\nTo deactivate:")
    print("  deactivate")

if __name__ == "__main__":
    main()

