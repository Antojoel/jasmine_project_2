import streamlit as st
import sys
import os

# Add the current directory to the system path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the web interface
import web_interface

# Run the web interface
web_interface.run_web_interface()