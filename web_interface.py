import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import os
import sys
import importlib.util

def run_web_interface():
    """
    Creates a web interface for the Allstate Auto Insurance Claim Settlement System
    where users can enter a policy number and upload a car damage image.
    The system processes the claim and displays the results on the same page.
    """
    # Import the main module directly using the module path
    main_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main2.py")
    module_name = "main_module"
    
    # Use importlib to import MAIN2.PY as a module regardless of its name
    spec = importlib.util.spec_from_file_location(module_name, main_file_path)
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    # Now use the imported module
    process_claim = main_module.process_claim
    ClaimStatus = main_module.ClaimStatus
    
    # Set up the Streamlit page
    st.set_page_config(
        page_title="Allstate Auto Insurance Claim System",
        page_icon="🚗",
        layout="wide"
    )
    
    # Header
    st.title("Allstate Auto Insurance Claim Settlement System")
    st.markdown("### Submit your claim by entering your policy number and uploading an image of the damage")
    
    # Create a two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Input form
        st.subheader("Claim Submission")
        policy_number = st.text_input("Enter your policy number (e.g., 999 876 543)")
        uploaded_file = st.file_uploader("Upload an image of the car damage", type=["jpg", "jpeg", "png"])
        
        # Submit button
        submit_button = st.button("Submit Claim")
        
        # Display a sample policy for testing
        with st.expander("Need a sample policy number?"):
            st.write("For testing, you can use the following policy numbers:")
            st.write("- Valid policy: 999 876 543")
            st.write("- Delinquent policy: 999 876 540")
            st.write("- Invalid policy: 111 222 333")
    
    # Process the claim when the submit button is clicked
    if submit_button and policy_number and uploaded_file:
        # Convert the uploaded image to base64
        image = Image.open(uploaded_file)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        damage_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Process the claim
        with st.spinner("Processing your claim... This may take a moment."):
            # Process the claim with actual image data
            result = process_claim(policy_number, damage_image_base64)
            
        # Display the results in the second column
        with col2:
            st.subheader("Claim Results")
            
            # Display the uploaded image
            st.image(image, caption="Uploaded Damage Image", width=400)
            
            # Display claim status with appropriate styling
            status = result["claim_status"]
            if status in [ClaimStatus.COMPLETED, ClaimStatus.PAYMENT_AUTHORIZED]:
                st.success(f"Claim Status: {status}")
            elif status in [ClaimStatus.DENIED_DELINQUENT, ClaimStatus.DENIED_INVALID_POLICY]:
                st.error(f"Claim Status: {status}")
            else:
                st.info(f"Claim Status: {status}")
            
            # Policy verification results
            st.markdown("#### Policy Verification")
            if result["policy_verified"]:
                st.write("✅ Policy verified")
            else:
                st.write("❌ Policy verification failed")
            
            st.write(result["policy_verification_result"])
            
            if result.get("delinquent", False):
                st.warning("⚠️ Policy is delinquent")
            
            # Damage assessment results if available
            if result["damage_assessment"] and result["policy_verified"] and not result.get("delinquent", False):
                st.markdown("#### Damage Assessment")
                
                damage_level = result["damage_assessment"].get("damage_level", "N/A")
                if damage_level == "minor":
                    damage_icon = "🟢"
                elif damage_level == "moderate":
                    damage_icon = "🟠"
                else:  # major
                    damage_icon = "🔴"
                
                st.write(f"{damage_icon} Damage Level: {damage_level.capitalize()}")
                st.write(f"Estimated Repair Cost: ${result['damage_assessment'].get('estimated_repair_cost', 0):.2f}")
                st.write(f"Recommendation: {result['damage_assessment'].get('recommendation', 'N/A')}")
                
                # Payment details if available
                if result["payment_amount"] > 0:
                    st.markdown("#### Payment Information")
                    st.write(f"Payment Amount: ${result['payment_amount']:.2f}")
                    
                    if result["payment_authorized"]:
                        st.success("✅ Payment Authorized")
                    else:
                        st.info("⏳ Payment Pending Authorization")
            
            # Message log
            st.markdown("#### Message Log")
            for i, message in enumerate(result["messages"], 1):
                st.write(f"{i}. {message}")
    
    elif submit_button:
        # Handle missing information
        if not policy_number:
            st.warning("Please enter your policy number")
        if not uploaded_file:
            st.warning("Please upload an image of the car damage")
    
    # Information panel at the bottom
    st.markdown("---")
    st.markdown("### About This System")
    st.write("""
    This system uses AI agents to process auto insurance claims:
    1. **Policy Verification Agent**: Verifies your policy number and checks for delinquency
    2. **Damage Assessment Agent**: Analyzes the damage image and classifies severity
    3. **Claim Processing Agent**: Calculates payment and sends notifications
    
    The system is built using Langchain and Langgraph frameworks with OpenAI's models.
    """)