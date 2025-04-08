
import sys
import os
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import the web interface module
import web_interface
# Run the web interface
web_interface.run_web_interface()
                