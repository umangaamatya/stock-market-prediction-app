#!/usr/bin/env python3
"""
Installation script for Stock Market Prediction Application
Automates the setup process for users
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

class StockMarketInstaller:
    """Automated installer for the stock market prediction application"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / "venv"
        self.config_path = self.project_root / ".env"
        
    def run_installation(self):
        """Run the complete installation process"""
        print("🏦 Stock Market Prediction App - Installation")
        print("=" * 50)
        
        try:
            # Step 1: Check system requirements
            print("📋 Checking system requirements...")
            if not self.check_system_requirements():
                return False
            
            # Step 2: Create virtual environment
            print("🐍 Creating Python virtual environment...")
            if not self.create_virtual_environment():
                return False
            
            # Step 3: Install dependencies
            print("📦 Installing Python dependencies...")
            if not self.install_dependencies():
                return False
            
            # Step 4: Setup configuration
            print("⚙️ Setting up configuration...")
            self.setup_configuration()
            
            # Step 5: Initialize directories
            print("📁 Creating necessary directories...")
            self.create_directories()
            
            # Step 6: Database setup instructions
            print("🗄️ Database setup instructions...")
            self.show_database_setup()
            
            # Step 7: API keys setup
            print("🔑 API keys setup instructions...")
            self.show_api_setup()
            
            print("\n✅ Installation completed successfully!")
            self.show_next_steps()
            return True
            
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    def check_system_requirements(self):
        """Check if system meets requirements"""
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required")
            print(f"Current version: {sys.version}")
            return False
        print("✅ Python version OK")
        
        # Check if PostgreSQL is available
        try:
            result = subprocess.run(['psql', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ PostgreSQL found")
            else:
                print("⚠️ PostgreSQL not found in PATH")
                print("Please install PostgreSQL before continuing")
        except FileNotFoundError:
            print("⚠️ PostgreSQL not found")
            print("Please install PostgreSQL before continuing")
        
        # Check if pip is available
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                          check=True, capture_output=True)
            print("✅ pip available")
        except subprocess.CalledProcessError:
            print("❌ pip not available")
            return False
        
        return True
    
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        try:
            if self.venv_path.exists():
                print("Virtual environment already exists, removing old one...")
                shutil.rmtree(self.venv_path)
            
            # Create virtual environment
            subprocess.run([sys.executable, '-m', 'venv', str(self.venv_path)], 
                          check=True)
            
            # Determine activation script path
            if os.name == 'nt':  # Windows
                activate_path = self.venv_path / 'Scripts' / 'activate'
                pip_path = self.venv_path / 'Scripts' / 'pip'
            else:  # Unix/Linux/macOS
                activate_path = self.venv_path / 'bin' / 'activate'
                pip_path = self.venv_path / 'bin' / 'pip'
            
            print(f"✅ Virtual environment created at {self.venv_path}")
            print(f"Activation script: {activate_path}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self):
        """Install Python dependencies"""
        try:
            # Determine pip path
            if os.name == 'nt':  # Windows
                pip_path = self.venv_path / 'Scripts' / 'pip'
            else:  # Unix/Linux/macOS
                pip_path = self.venv_path / 'bin' / 'pip'
            
            # Upgrade pip first
            subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], 
                          check=True)
            
            # Install requirements
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                subprocess.run([str(pip_path), 'install', '-r', str(requirements_file)], 
                              check=True)
                print("✅ Dependencies installed successfully")
            else:
                print("⚠️ requirements.txt not found")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_configuration(self):
        """Setup configuration file"""
        template_path = self.project_root / '.env.template'
        
        if template_path.exists() and not self.config_path.exists():
            shutil.copy(template_path, self.config_path)
            print(f"✅ Configuration template copied to {self.config_path}")
            print("📝 Please edit .env file with your settings")
        else:
            print("⚠️ Configuration file already exists or template not found")
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw',
            'data/processed', 
            'models/saved_models',
            'logs'
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
    
    def show_database_setup(self):
        """Show database setup instructions"""
        print("\n📊 DATABASE SETUP INSTRUCTIONS")
        print("-" * 40)
        print("1. Install PostgreSQL if not already installed:")
        print("   • Windows: Download from https://www.postgresql.org/download/windows/")
        print("   • macOS: brew install postgresql")
        print("   • Ubuntu: sudo apt-get install postgresql postgresql-contrib")
        print()
        print("2. Create database and user:")
        print("   sudo -u postgres psql")
        print("   CREATE DATABASE stock_market_db;")
        print("   CREATE USER stock_user WITH PASSWORD 'your_secure_password';")
        print("   GRANT ALL PRIVILEGES ON DATABASE stock_market_db TO stock_user;")
        print("   \\q")
        print()
        print("3. Update .env file with your database credentials")
    
    def show_api_setup(self):
        """Show API keys setup instructions"""
        print("\n🔑 API KEYS SETUP")
        print("-" * 40)
        print("To get full functionality, sign up for these free API keys:")
        print()
        print("1. News API (for sentiment analysis):")
        print("   • Go to: https://newsapi.org/")
        print("   • Sign up for a free account")
        print("   • Get your API key")
        print("   • Add to .env file: NEWS_API_KEY=your_key_here")
        print()
        print("2. Alpha Vantage (optional, for additional stock data):")
        print("   • Go to: https://www.alphavantage.co/support/#api-key")
        print("   • Get your free API key")
        print("   • Add to .env file: ALPHA_VANTAGE_API_KEY=your_key_here")
        print()
        print("Note: The app will work with just Yahoo Finance data (no API key needed)")
    
    def show_next_steps(self):
        """Show next steps after installation"""
        print("\n🚀 NEXT STEPS")
        print("-" * 40)
        print("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f"   {self.venv_path}\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print(f"   source {self.venv_path}/bin/activate")
        
        print("\n2. Configure your settings:")
        print(f"   Edit {self.config_path} with your database and API credentials")
        
        print("\n3. Set up the database:")
        print("   Follow the database setup instructions above")
        
        print("\n4. Run the ETL pipeline to load initial data:")
        print("   python -m etl.etl_pipeline --validate")
        print("   python -m etl.etl_pipeline --mode full")
        
        print("\n5. Start the web application:")
        print("   streamlit run web_app/app.py")
        
        print("\n6. Open your browser and go to:")
        print("   http://localhost:8501")
        
        print("\n📚 For detailed instructions, see the User Manual in docs/USER_MANUAL.md")

def main():
    """Main installation function"""
    installer = StockMarketInstaller()
    success = installer.run_installation()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
