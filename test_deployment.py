"""
Test script to verify the app's ability to handle large model files
before deploying to Streamlit Cloud.
"""
import os
import sys
import joblib
import tempfile
import gdown
from pathlib import Path
import time
import argparse
import requests

# Model paths
MODEL_DIR = "model"
SELECTOR_PATH = os.path.join(MODEL_DIR, 'feature_selector.joblib')

def check_environment():
    """Check if the environment is set up correctly"""
    print("==== Environment Check ====")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check required directories
    if not os.path.exists(MODEL_DIR):
        print(f"⚠️ Warning: Model directory '{MODEL_DIR}' does not exist")
        create = input("Create it? (y/n): ")
        if create.lower() == 'y':
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"✅ Created model directory: {MODEL_DIR}")
    else:
        print(f"✅ Model directory exists: {MODEL_DIR}")
    
    # Check for .streamlit directory
    if not os.path.exists('.streamlit'):
        print("⚠️ Warning: .streamlit directory does not exist")
        create = input("Create it? (y/n): ")
        if create.lower() == 'y':
            os.makedirs('.streamlit', exist_ok=True)
            print(f"✅ Created .streamlit directory")
    else:
        print(f"✅ .streamlit directory exists")
    
    # Check for config.toml
    if not os.path.exists('.streamlit/config.toml'):
        print("⚠️ Warning: .streamlit/config.toml does not exist")
    else:
        print(f"✅ .streamlit/config.toml exists")
    
    # Check for secrets.toml
    if not os.path.exists('.streamlit/secrets.toml'):
        print("⚠️ Warning: .streamlit/secrets.toml does not exist")
        if os.path.exists('.streamlit/secrets_example.toml'):
            create = input("Create from example? (y/n): ")
            if create.lower() == 'y':
                with open('.streamlit/secrets_example.toml', 'r') as src:
                    content = src.read()
                with open('.streamlit/secrets.toml', 'w') as dst:
                    dst.write(content)
                print(f"✅ Created .streamlit/secrets.toml from example")
    else:
        print(f"✅ .streamlit/secrets.toml exists")
    
    # Check required Python packages
    try:
        import streamlit
        print(f"✅ streamlit is installed (version {streamlit.__version__})")
    except ImportError:
        print("❌ streamlit is not installed")
    
    try:
        import pandas
        print(f"✅ pandas is installed (version {pandas.__version__})")
    except ImportError:
        print("❌ pandas is not installed")
    
    try:
        import numpy
        print(f"✅ numpy is installed (version {numpy.__version__})")
    except ImportError:
        print("❌ numpy is not installed")
    
    try:
        import gdown
        print(f"✅ gdown is installed")
    except ImportError:
        print("❌ gdown is not installed")
    
    print("\n")

def test_model_loading():
    """Test direct loading of the model file"""
    print(f"Testing direct loading of {SELECTOR_PATH}...")
    if os.path.exists(SELECTOR_PATH):
        size_mb = Path(SELECTOR_PATH).stat().st_size / (1024 * 1024)
        print(f"File exists, size: {size_mb:.2f} MB")
        
        # Check if file size exceeds Streamlit limit
        if size_mb > 240:
            print(f"⚠️ Warning: File size ({size_mb:.2f} MB) exceeds Streamlit's recommended limit (240 MB)")
            print("   You will need to use the Google Drive workaround for deployment")
        
        try:
            start_time = time.time()
            model = joblib.load(SELECTOR_PATH)
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print(f"❌ File not found: {SELECTOR_PATH}")
        return False

def get_gdrive_file_id():
    """Get Google Drive file ID from secrets file or user input"""
    secrets_path = Path('.streamlit/secrets.toml')
    file_id = None
    
    if secrets_path.exists():
        # Parse the secrets file manually to get the file ID
        with open(secrets_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith('SELECTOR_GDRIVE_URL'):
                # Extract the file ID from the line
                parts = line.strip().split('=')
                if len(parts) == 2:
                    file_id = parts[1].strip().strip('"').strip("'")
        
        if not file_id or file_id == "YOUR_FILE_ID_HERE":
            print("No valid file ID found in secrets.toml.")
            file_id = input("Enter your Google Drive file ID to test: ")
    else:
        print("Warning: No secrets.toml file found. Please create one from the example.")
        file_id = input("Enter your Google Drive file ID to test: ")
    
    return file_id

def check_file_accessibility(file_id):
    """Check if a Google Drive file is publicly accessible"""
    print(f"Checking if file is publicly accessible...")
    
    # Try to get file info
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            print(f"✅ File appears to be accessible (status code: {response.status_code})")
            return True
        else:
            print(f"❌ File might not be publicly accessible (status code: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error checking file accessibility: {e}")
        return False

def test_gdrive_loading():
    """Test loading the model from Google Drive using the gdown library"""
    file_id = get_gdrive_file_id()
    
    if not file_id:
        print("No file ID provided, skipping Google Drive test.")
        return False
    
    # First check if the file is accessible
    if not check_file_accessibility(file_id):
        print("Make sure the file is shared with 'Anyone with the link' on Google Drive")
    
    # Create a temporary file to download to
    print(f"Testing download from Google Drive with file ID: {file_id}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as temp_file:
            temp_path = temp_file.name
        
        gdrive_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading to {temp_path}...")
        start_time = time.time()
        output = gdown.download(gdrive_url, temp_path, quiet=False, fuzzy=True)
        
        if output:
            download_time = time.time() - start_time
            print(f"✅ File downloaded successfully in {download_time:.2f} seconds")
            
            # Get file size
            size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
            print(f"Downloaded file size: {size_mb:.2f} MB")
            
            # Try to load the model
            try:
                start_time = time.time()
                model = joblib.load(temp_path)
                load_time = time.time() - start_time
                print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
                return True
            except Exception as e:
                print(f"❌ Error loading downloaded model: {e}")
                return False
            finally:
                # Clean up
                try:
                    os.unlink(temp_path)
                    print("✅ Temporary file cleaned up")
                except:
                    print("❌ Failed to clean up temporary file")
        else:
            print("❌ Failed to download file from Google Drive")
            return False
    except Exception as e:
        print(f"❌ Error in Google Drive test: {e}")
        return False

def verify_streamlit_deployment():
    """Verify that the app can be deployed to Streamlit Cloud"""
    print("==== Streamlit Deployment Verification ====")
    
    # Check if we have a requirements.txt file
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt file not found")
    else:
        print("✅ requirements.txt file exists")
        
        # Check if gdown is in requirements.txt
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            if 'gdown' in requirements:
                print("✅ gdown is listed in requirements.txt")
            else:
                print("❌ gdown not found in requirements.txt")
    
    # Check if we have an app.py file
    if not os.path.exists('app.py'):
        print("❌ app.py file not found")
    else:
        print("✅ app.py file exists")
    
    # Check that we have the correct code in app.py to handle deployment
    with open('app.py', 'r') as f:
        content = f.read()
        if 'DEPLOYMENT_ENV' in content and 'gdown.download' in content:
            print("✅ app.py appears to be configured for deployment")
        else:
            print("❌ app.py may not be properly configured for deployment")
    
    print("\n")

def show_deployment_steps():
    """Show deployment steps as a reminder"""
    print("==== Deployment Steps ====")
    print("1. Upload your large model file to Google Drive")
    print("2. Make the file publicly accessible (Share > Anyone with the link)")
    print("3. Get the file ID from the Google Drive URL")
    print("4. Create a GitHub repository and push your code")
    print("5. On Streamlit Cloud:")
    print("   - Create a new app")
    print("   - Connect to your GitHub repository")
    print("   - Add the following to secrets:")
    print("     SELECTOR_GDRIVE_URL = 'YOUR_FILE_ID'")
    print("   - Add the following to environment variables:")
    print("     STREAMLIT_DEPLOYMENT = cloud")
    print("   - Deploy")
    print("\n")

def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description='Test Streamlit deployment with large model files')
    parser.add_argument('--env-check', action='store_true', help='Check environment setup')
    parser.add_argument('--local-test', action='store_true', help='Test local model loading')
    parser.add_argument('--gdrive-test', action='store_true', help='Test Google Drive model loading')
    parser.add_argument('--deploy-check', action='store_true', help='Check deployment configuration')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # If no arguments provided, run all tests
    if not any(vars(args).values()):
        args.all = True
    
    if args.all or args.env_check:
        check_environment()
    
    if args.all or args.local_test:
        direct_test = test_model_loading()
    else:
        direct_test = None
    
    if args.all or args.gdrive_test:
        gdrive_test = test_gdrive_loading()
    else:
        gdrive_test = None
    
    if args.all or args.deploy_check:
        verify_streamlit_deployment()
    
    if args.all:
        show_deployment_steps()
    
    if args.all or args.local_test or args.gdrive_test:
        print("\n==== Test Results ====")
        if direct_test is not None:
            print(f"Direct loading: {'✅ PASSED' if direct_test else '❌ FAILED'}")
        if gdrive_test is not None:
            print(f"Google Drive loading: {'✅ PASSED' if gdrive_test else '❌ FAILED'}")
        
        if direct_test is not None and gdrive_test is not None:
            if direct_test and gdrive_test:
                print("\n✅ All tests passed! Your app should work both locally and in Streamlit Cloud.")
            elif direct_test:
                print("\n⚠️ Direct loading works but Google Drive loading failed. Fix Google Drive setup for cloud deployment.")
            elif gdrive_test:
                print("\n⚠️ Google Drive loading works but direct loading failed. This is unusual and may indicate file corruption.")
            else:
                print("\n❌ All tests failed. Please check your model files and Google Drive setup.")
        
        print("\nNext steps:")
        print("1. Run 'streamlit run app.py' to test the app locally")
        print("2. Deploy to Streamlit Cloud and configure secrets")
        print("3. Set STREAMLIT_DEPLOYMENT=cloud in environment variables")

if __name__ == "__main__":
    main()