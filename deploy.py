#!/usr/bin/env python3
"""
Helper script for deploying the application to Streamlit Cloud
"""
import os
import sys
import argparse
import webbrowser
import time
from pathlib import Path

STREAMLIT_CLOUD_URL = "https://share.streamlit.io"
MODEL_DIR = "model"

def check_git_repo():
    """Check if this is a git repository"""
    if not os.path.exists('.git'):
        print("❌ This doesn't appear to be a git repository")
        print("You need to initialize a git repository first:")
        print("   git init")
        print("   git add .")
        print("   git commit -m 'Initial commit'")
        return False
    return True

def get_remote_urls():
    """Get the git remote URLs"""
    if not check_git_repo():
        return []
    
    try:
        import subprocess
        result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except:
        return []

def check_deployment_readiness():
    """Check if everything is ready for deployment"""
    score = 0
    max_score = 7
    
    print("==== Deployment Readiness Check ====")
    
    # Check 1: app.py exists
    if os.path.exists('app.py'):
        print("✅ app.py exists")
        score += 1
    else:
        print("❌ app.py not found")
    
    # Check 2: requirements.txt exists and has required packages
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            content = f.read().lower()
            required_packages = ['streamlit', 'pandas', 'numpy', 'joblib', 'gdown']
            missing = [pkg for pkg in required_packages if pkg not in content]
            
            if not missing:
                print("✅ requirements.txt exists with all required packages")
                score += 1
            else:
                print(f"⚠️ requirements.txt is missing these packages: {', '.join(missing)}")
    else:
        print("❌ requirements.txt not found")
    
    # Check 3: .streamlit directory with config
    if os.path.exists('.streamlit/config.toml'):
        print("✅ .streamlit/config.toml exists")
        score += 1
    else:
        print("❌ .streamlit/config.toml not found")
    
    # Check 4: secrets file or example
    if os.path.exists('.streamlit/secrets.toml'):
        print("✅ .streamlit/secrets.toml exists")
        score += 1
    elif os.path.exists('.streamlit/secrets_example.toml'):
        print("⚠️ .streamlit/secrets_example.toml exists but no secrets.toml")
        print("   This is okay for deployment, as secrets will be set in Streamlit Cloud")
        score += 0.5
    else:
        print("❌ No secrets configuration found")
    
    # Check 5: Model files exist
    small_models = ['multi_output_model.joblib', 'label_encoders.joblib', 
                   'feature_cols.joblib', 'inverse_maps.joblib']
    missing_models = [model for model in small_models 
                     if not os.path.exists(os.path.join(MODEL_DIR, model))]
    
    if not missing_models:
        print("✅ All required model files exist")
        score += 1
    else:
        print(f"❌ Missing model files: {', '.join(missing_models)}")
    
    # Check 6: .gitignore with appropriate entries
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r') as f:
            content = f.read().lower()
            if 'feature_selector.joblib' in content or (MODEL_DIR + '/feature_selector.joblib').lower() in content:
                print("✅ .gitignore includes large model file")
                score += 1
            else:
                print("⚠️ .gitignore should include feature_selector.joblib")
                print("   This is recommended to avoid pushing large files to GitHub")
    else:
        print("⚠️ No .gitignore file found")
    
    # Check 7: Git repository
    if check_git_repo():
        print("✅ Git repository initialized")
        
        remotes = get_remote_urls()
        if remotes:
            print(f"✅ Git remote configured: {remotes[0] if remotes else 'None'}")
            score += 1
        else:
            print("⚠️ No git remote configured")
            print("   You'll need to add a remote repository:")
            print("   git remote add origin <repository-url>")
    else:
        print("❌ Not a git repository")
    
    # Display final score
    print(f"\nReadiness score: {score}/{max_score} ({score/max_score*100:.1f}%)")
    
    if score >= 6:
        print("✅ Your project is ready for deployment!")
    elif score >= 4:
        print("⚠️ Your project might be ready for deployment, but there are some issues to address.")
    else:
        print("❌ Your project is not ready for deployment. Please address the issues above.")
    
    return score, max_score

def show_checklist():
    """Display deployment checklist"""
    print("\n==== Deployment Checklist ====")
    print("Before deploying to Streamlit Cloud, please confirm:")
    
    print("\n1. Large Model File:")
    print("   ☐ I've uploaded feature_selector.joblib to Google Drive")
    print("   ☐ I've shared the file with 'Anyone with the link'")
    print("   ☐ I have the file ID from the Google Drive URL")
    
    print("\n2. GitHub Repository:")
    print("   ☐ I've created a GitHub repository")
    print("   ☐ I've added the remote to my local repository")
    print("   ☐ I've pushed all my code to GitHub (excluding large model file)")
    
    print("\n3. Streamlit Cloud Configuration:")
    print("   ☐ I've created a Streamlit Cloud account")
    print("   ☐ I'm ready to set the SELECTOR_GDRIVE_URL secret")
    print("   ☐ I'm ready to set the STREAMLIT_DEPLOYMENT environment variable")
    
    print("\nIf you've checked all of these, you're ready to proceed to deployment!")

def open_streamlit_cloud():
    """Open Streamlit Cloud in browser"""
    print("Opening Streamlit Cloud in your browser...")
    webbrowser.open(STREAMLIT_CLOUD_URL)
    print("Please log in and follow these steps:")
    print("1. Click 'New app'")
    print("2. Connect to your GitHub repository")
    print("3. Configure the app:")
    print("   - Main file path: app.py")
    print("   - Python version: 3.9 (or newer)")
    print("   - Add environment variable: STREAMLIT_DEPLOYMENT = cloud")
    print("   - Add secret: SELECTOR_GDRIVE_URL = <your-file-id>")
    print("4. Click 'Deploy!'")

def main():
    """Main deployment helper function"""
    parser = argparse.ArgumentParser(description="Streamlit Deployment Helper")
    parser.add_argument('--check', action='store_true', help="Check deployment readiness")
    parser.add_argument('--checklist', action='store_true', help="Show deployment checklist")
    parser.add_argument('--deploy', action='store_true', help="Open Streamlit Cloud for deployment")
    
    args = parser.parse_args()
    
    # If no args, show all
    if not any(vars(args).values()):
        score, max_score = check_deployment_readiness()
        print("\n" + "="*50 + "\n")
        show_checklist()
        
        if score >= 4:  # Only suggest deployment if score is reasonable
            print("\n" + "="*50 + "\n")
            proceed = input("Would you like to proceed to deployment? (y/n): ")
            if proceed.lower() == 'y':
                open_streamlit_cloud()
    else:
        if args.check:
            check_deployment_readiness()
        
        if args.checklist:
            show_checklist()
        
        if args.deploy:
            open_streamlit_cloud()

if __name__ == "__main__":
    main() 