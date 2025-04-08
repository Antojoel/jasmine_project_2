import os
import base64
import json
import re
from typing import Dict, TypedDict, Annotated, Sequence, List, Union, Any, Literal, cast
from enum import Enum
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


# Setup your API key
# Replace with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = "key here"


# Models
gpt4_vision = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4096)
gpt4 = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=2048)


# Define the possible status values as an Enum
class ClaimStatus(str, Enum):
    SUBMITTED = "submitted"
    POLICY_VERIFICATION = "policy_verification"
    DENIED_INVALID_POLICY = "denied_invalid_policy"
    DENIED_DELINQUENT = "denied_delinquent"
    DAMAGE_ASSESSMENT = "damage_assessment"
    PAYMENT_PROCESSING = "payment_processing"
    PAYMENT_AUTHORIZED = "payment_authorized"
    COMPLETED = "completed"


# Define State Type
class ClaimState(TypedDict):
    policy_number: str
    damage_image: str
    policy_verified: bool
    policy_verification_result: str
    delinquent: bool
    damage_assessment: Dict
    claim_status: str
    payment_amount: float
    payment_authorized: bool
    notification_sent: bool
    messages: List[str]  # To store messages for the user


# ----- Policy Verification Agent -----

class PolicySchema(BaseModel):
    policy_valid: bool = Field(description="Whether the policy number is valid")
    delinquent: bool = Field(description="Whether the insured is delinquent on payments")
    verification_message: str = Field(description="Message explaining verification result")


@tool
def verify_policy(policy_number: str) -> Dict:
    """
    Verifies if the policy number is valid and if the insured is not delinquent.
    
    Args:
        policy_number: The policy number to verify
        
    Returns:
        A dictionary with verification results
    """
    # Hardcoded policy data for this example
    # In a real implementation, this would query a database
    valid_policies = ["999 876 543", "123 456 789", "987 654 321","999 876 540"]
    
    # Check if policy is valid
    if policy_number.strip() in valid_policies:
        result = {
            "policy_valid": True,
            "delinquent": False,  # Assuming not delinquent for this valid policy
            "verification_message": "Policy verified successfully. The policy is active and payments are up to date."
        }
    else:
        result = {
            "policy_valid": False,
            "delinquent": False,  # Not relevant if policy is invalid
            "verification_message": f"Policy verification failed. The policy number {policy_number} is not found in our records."
        }
    
    return result


@tool
def check_delinquency(policy_number: str) -> Dict:
    """
    Checks if the policy holder is delinquent on payments.
    
    Args:
        policy_number: The policy number to check
        
    Returns:
        A dictionary with delinquency check results
    """
    # Hardcoded delinquency check for this example
    # In a real implementation, this would query a database
    
    # For demonstration, we'll make policy numbers ending in 0 delinquent
    if policy_number.strip().endswith('0'):
        result = {
            "delinquent": True,
            "message": "The policy holder is delinquent on payments. Claim processing is denied."
        }
    else:
        result = {
            "delinquent": False,
            "message": "The policy holder is current on all payments."
        }
    
    return result


class DamageSchema(BaseModel):
    damage_level: str = Field(description="Level of damage: minor, moderate, or major")
    damage_description: str = Field(description="Detailed description of the damage")
    estimated_repair_cost: float = Field(description="Estimated cost to repair the damage")
    recommendation: str = Field(description="Recommendation for the claim")


# Helper function for testing
def simulate_damage_assessment(damage_level: str) -> Dict:
    """
    Simulates damage assessment for testing purposes.
    
    Args:
        damage_level: The level of damage to simulate (minor, moderate, major)
        
    Returns:
        A simulated damage assessment dictionary
    """
    if damage_level == "minor":
        return {
            "damage_level": "minor",
            "damage_description": "Light scratches and minor dent on the front bumper.",
            "estimated_repair_cost": 800.00,
            "recommendation": "Approve claim for minor cosmetic repair."
        }
    elif damage_level == "moderate":
        return {
            "damage_level": "moderate",
            "damage_description": "Damaged front bumper with dents and broken headlight.",
            "estimated_repair_cost": 2200.00,
            "recommendation": "Approve claim for standard repair procedure."
        }
    else:  # major
        return {
            "damage_level": "major",
            "damage_description": "Severe front-end damage with bumper detachment, broken headlights, and hood damage.",
            "estimated_repair_cost": 5800.00,
            "recommendation": "Approve claim for extensive repair. Possible frame damage should be inspected."
        }


# Function to calculate payment amount based on damage assessment
@tool
def calculate_payment(damage_assessment: Dict) -> Dict:
    """
    Calculates the payment amount based on the damage assessment.
    
    Args:
        damage_assessment: Dictionary containing damage assessment results
        
    Returns:
        A dictionary with payment calculation results
    """
    damage_level = damage_assessment.get("damage_level", "moderate")
    estimated_cost = damage_assessment.get("estimated_repair_cost", 1000.0)
    
    # Apply coverage and deductible based on damage level
    # In a real system, this would consider the specific policy details
    deductible = 500.0  # Standard deductible
    
    if damage_level == "minor":
        coverage_percentage = 0.8  # 80% coverage for minor damage
    elif damage_level == "moderate":
        coverage_percentage = 0.9  # 90% coverage for moderate damage
    else:  # major
        coverage_percentage = 0.95  # 95% coverage for major damage
    
    # Calculate payment amount
    payment_amount = max(0, (estimated_cost * coverage_percentage) - deductible)
    
    return {
        "payment_amount": round(payment_amount, 2),
        "deductible_applied": deductible,
        "coverage_percentage": coverage_percentage,
        "message": f"Approved payment of ${payment_amount:.2f} for {damage_level} damage repair."
    }


@tool
def assess_damage(damage_image: str, test_mode: bool = False, test_damage_level: str = None) -> Dict:
    """
    Assesses the damage shown in an image using AI vision capabilities.
    
    Args:
        damage_image: Base64 encoded image of the car damage
        test_mode: If True, skip actual API call and use test data
        test_damage_level: The damage level to use in test mode
        
    Returns:
        A dictionary with damage assessment results
    """
    # If in test mode, return simulated assessment
    if test_mode and test_damage_level:
        return simulate_damage_assessment(test_damage_level)
    
    # Use GPT-4 Vision to analyze the image
    # First, we need to properly format the image for the API
    damage_prompt = PromptTemplate.from_template(
        """You are an expert auto insurance damage assessor. Analyze the image of car damage and provide:
        1. Damage level (minor, moderate, or major)
        2. Detailed description of the visible damage
        3. Estimated repair cost range
        4. Recommendation for claim approval
        
        Format your response as a JSON object with the following fields:
        - damage_level: string (minor, moderate, or major)
        - damage_description: string
        - estimated_repair_cost: float (average value of the estimated range)
        - recommendation: string
        
        Image of the damage:
        {image}
        """
    )
    
    # Image is already provided as base64, so we'll use it directly
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": damage_prompt.format(image="")},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{damage_image}"}}
            ]
        )
    ]
    
    # Get response from vision model
    response = gpt4_vision.invoke(messages)
    
    # Parse the response to extract the JSON
    content = response.content
    json_match = re.search(r'{.*}', content, re.DOTALL)
    if json_match:
        try:
            assessment_result = json.loads(json_match.group(0))
            return assessment_result
        except json.JSONDecodeError:
            # Fallback for demonstration purposes
            return {
                "damage_level": "moderate",
                "damage_description": "Front bumper damage with visible scratches and dents",
                "estimated_repair_cost": 1200.00,
                "recommendation": "Approve claim for standard repair procedure"
            }
    else:
        # Fallback for demonstration purposes
        return {
            "damage_level": "moderate",
            "damage_description": "Front bumper damage with visible scratches and dents",
            "estimated_repair_cost": 1200.00,
            "recommendation": "Approve claim for standard repair procedure"
        }


@tool
def authorize_payment(payment_amount: float, damage_level: str) -> Dict:
    """
    Authorizes the payment for a claim.
    
    Args:
        payment_amount: The amount to be paid
        damage_level: Level of damage (minor, moderate, major)
        
    Returns:
        A dictionary with payment authorization results
    """
    # In a real system, this would interact with payment systems
    # For demonstration, we'll approve all payments under a threshold
    max_auto_approval = 5000.0
    
    if payment_amount <= max_auto_approval:
        result = {
            "authorized": True,
            "authorization_id": f"AUTH-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "message": f"Payment of ${payment_amount:.2f} authorized automatically."
        }
    else:
        # Simulating a more complex authorization process for large amounts
        if damage_level == "major":
            result = {
                "authorized": True,
                "authorization_id": f"AUTH-{datetime.now().strftime('%Y%m%d%H%M%S')}-REVIEW",
                "message": f"Payment of ${payment_amount:.2f} authorized after review."
            }
        else:
            result = {
                "authorized": False,
                "message": f"Payment of ${payment_amount:.2f} requires additional review."
            }
    
    return result


# Define schema for notification
class NotificationData(BaseModel):
    recipient: str = Field(description="The recipient of the notification (policy number)")
    claim_status: str = Field(description="The current status of the claim")
    message: str = Field(description="The notification message")

@tool
def send_notification(recipient: str, claim_status: str, message: str) -> Dict:
    """
    Sends a notification to the insured person.
    
    Args:
        recipient: The policy number of the recipient
        claim_status: The current status of the claim
        message: The notification message
        
    Returns:
        A dictionary with notification results
    """
    # In a real system, this would send an email or SMS
    # For demonstration, we'll just return the notification details
    
    return {
        "sent": True,
        "recipient": recipient,
        "claim_status": claim_status,
        "message": message,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


# For damage assessment agent, also need to update the tool invocation
def damage_assessment_agent(state: ClaimState, test_mode: bool = False, test_damage_level: str = None) -> ClaimState:
    """
    Agent that assesses the damage based on the image.
    
    Args:
        state: The current state of the claim
        test_mode: If True, use test data instead of calling the API
        test_damage_level: The damage level to use in test mode
    """
    # If the policy wasn't verified or is delinquent, we don't proceed
    if not state["policy_verified"] or state.get("delinquent", False):
        return state
    
    # Assess the damage from the image
    if test_mode and test_damage_level:
        damage_assessment_result = simulate_damage_assessment(test_damage_level)
    else:
        # Fix: Pass arguments as a dictionary
        damage_assessment_result = assess_damage.invoke({
            "damage_image": state["damage_image"],
            "test_mode": False,
            "test_damage_level": None
        })
    
    state["damage_assessment"] = damage_assessment_result
    
    # Calculate payment amount
    payment_calculation = calculate_payment.invoke({"damage_assessment": damage_assessment_result})
    state["payment_amount"] = payment_calculation["payment_amount"]
    
    # Add messages
    damage_message = f"Damage assessment: {damage_assessment_result['damage_level']} damage. {damage_assessment_result['damage_description']}"
    payment_message = payment_calculation["message"]
    
    state["messages"].append(damage_message)
    state["messages"].append(payment_message)
    
    # Move to payment processing
    state["claim_status"] = ClaimStatus.PAYMENT_PROCESSING
    
    return state


# For policy verification agent
# For policy verification agent
def policy_verification_agent(state: ClaimState) -> ClaimState:
    """Agent that verifies the policy and checks for delinquency."""
    policy_number = state["policy_number"]
    
    # Step 1: Verify if the policy is valid
    verification_result = verify_policy.invoke({"policy_number": policy_number})
    policy_verified = verification_result["policy_valid"]
    
    # Update state with verification results
    state["policy_verified"] = policy_verified
    state["policy_verification_result"] = verification_result["verification_message"]
    state["messages"].append(verification_result["verification_message"])
    
    # If policy is valid, check for delinquency
    if policy_verified:
        delinquency_result = check_delinquency.invoke({"policy_number": policy_number})
        state["delinquent"] = delinquency_result["delinquent"]
        state["messages"].append(delinquency_result["message"])
        
        if delinquency_result["delinquent"]:
            # CHANGE THIS PART: Currently it doesn't properly update the status for delinquent policies
            state["claim_status"] = ClaimStatus.DENIED_DELINQUENT
            # Send denial notification
            notification = send_notification.invoke({
                "recipient": policy_number,
                "claim_status": str(ClaimStatus.DENIED_DELINQUENT),
                "message": "Your claim has been denied because your policy is delinquent. Please contact customer service."
            })
            state["notification_sent"] = True
            state["messages"].append(notification["message"])
        else:
            state["claim_status"] = ClaimStatus.DAMAGE_ASSESSMENT
    else:
        state["claim_status"] = ClaimStatus.DENIED_INVALID_POLICY
        # Send denial notification
        notification = send_notification.invoke({
            "recipient": policy_number,
            "claim_status": str(ClaimStatus.DENIED_INVALID_POLICY),
            "message": "Your claim has been denied because the policy number is invalid. Please check and resubmit."
        })
        state["notification_sent"] = True
        state["messages"].append(notification["message"])
    
    return state

# For claim processing agent
def claim_processing_agent(state: ClaimState) -> ClaimState:
    """Agent that processes the claim and authorizes payment."""
    # If we're not in payment processing, don't proceed
    if state["claim_status"] != ClaimStatus.PAYMENT_PROCESSING:
        return state
    
    # Get the payment amount and damage level
    payment_amount = state["payment_amount"]
    damage_level = state["damage_assessment"]["damage_level"]
    
    # Authorize the payment
    authorization_result = authorize_payment.invoke({
        "payment_amount": payment_amount, 
        "damage_level": damage_level
    })
    state["payment_authorized"] = authorization_result["authorized"]
    
    # Add message
    state["messages"].append(authorization_result["message"])
    
    if authorization_result["authorized"]:
        state["claim_status"] = ClaimStatus.PAYMENT_AUTHORIZED
        
        # Send notification about approved payment
        notification = send_notification.invoke({
            "recipient": state["policy_number"],
            "claim_status": str(ClaimStatus.PAYMENT_AUTHORIZED),
            "message": f"Your claim has been approved. Payment of ${payment_amount:.2f} has been authorized for {damage_level} damage repair."
        })
        state["notification_sent"] = True
        state["messages"].append(notification["message"])
        
        # Mark as completed
        state["claim_status"] = ClaimStatus.COMPLETED
    else:
        # Notification about pending approval
        notification = send_notification.invoke({
            "recipient": state["policy_number"],
            "claim_status": str(state["claim_status"]),
            "message": f"Your claim requires additional review. We will notify you once a decision has been made."
        })
        state["notification_sent"] = True
        state["messages"].append(notification["message"])
    
    return state


# ----- Define the workflow/graph -----

def should_proceed_to_damage_assessment(state: ClaimState) -> Literal["damage_assessment_node", "end"]:
    """
    Determines if the workflow should proceed to damage assessment.
    """
    if state["claim_status"] == ClaimStatus.DAMAGE_ASSESSMENT:
        return "damage_assessment_node"
    return "end"


def should_proceed_to_claim_processing(state: ClaimState) -> Literal["claim_processing", "end"]:
    """
    Determines if the workflow should proceed to claim processing.
    """
    if state["claim_status"] == ClaimStatus.PAYMENT_PROCESSING:
        return "claim_processing"
    return "end"


def build_claim_processing_graph(test_mode: bool = False, test_damage_level: str = None):
    # Initialize the graph
    workflow = StateGraph(ClaimState)
    
    # Add nodes
    workflow.add_node("policy_verification", policy_verification_agent)
    
    # Use a custom function for damage assessment in test mode
    if test_mode:
        def test_damage_assessment_wrapper(state: ClaimState):
            return damage_assessment_agent(state, test_mode=False, test_damage_level=test_damage_level)
        workflow.add_node("damage_assessment_node", test_damage_assessment_wrapper)
    else:
        workflow.add_node("damage_assessment_node", damage_assessment_agent)
    
    workflow.add_node("claim_processing", claim_processing_agent)
    
    # Add edges - This is the correct way to define conditional edges
    workflow.add_conditional_edges(
        "policy_verification",
        should_proceed_to_damage_assessment,
        {
            "damage_assessment_node": "damage_assessment_node",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "damage_assessment_node",
        should_proceed_to_claim_processing,
        {
            "claim_processing": "claim_processing",
            "end": END
        }
    )
    
    # Add regular edge for the final step
    workflow.add_edge("claim_processing", END)
    
    # Set the entry point
    workflow.set_entry_point("policy_verification")
    
    return workflow.compile()


# ----- Function to run the workflow -----

def process_claim(policy_number: str, damage_image_base64: str, test_mode: bool = False, test_damage_level: str = None):
    """
    Processes a claim through the entire workflow.
    
    Args:
        policy_number: The policy number for the claim
        damage_image_base64: Base64 encoded image of the damage
        test_mode: If True, use test data instead of calling the API
        test_damage_level: The damage level to use in test mode
    
    Returns:
        The final state after processing the claim
    """
    # Initialize the graph
    graph = build_claim_processing_graph(test_mode, test_damage_level)
    
    # Initialize the state
    initial_state = ClaimState(
        policy_number=policy_number,
        damage_image=damage_image_base64,
        policy_verified=False,
        policy_verification_result="",
        delinquent=False,
        damage_assessment={},
        claim_status=ClaimStatus.SUBMITTED,
        payment_amount=0.0,
        payment_authorized=False,
        notification_sent=False,
        messages=[]
    )
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result


# ----- Helper functions for demonstration -----

def load_image_base64(image_path: str) -> str:
    """
    Loads an image and converts it to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def display_claim_results(result: ClaimState, scenario_name: str = None):
    """
    Displays the results of claim processing in the console.
    
    Args:
        result: The final state after processing the claim
        scenario_name: Optional name of the test scenario
    """
    border = "=" * 80
    
    print(border)
    print(f"ALLSTATE AUTO INSURANCE CLAIM RESULTS")
    if scenario_name:
        print(f"SCENARIO: {scenario_name}")
    print(border)
    
    print(f"\nCLAIM INFORMATION:")
    print(f"Policy Number: {result['policy_number']}")
    print(f"Claim Status: {result['claim_status']}")
    
    print(f"\nPOLICY VERIFICATION:")
    print(f"Policy Verified: {'Yes' if result['policy_verified'] else 'No'}")
    print(f"Result: {result['policy_verification_result']}")
    print(f"Delinquent: {'Yes' if result.get('delinquent', False) else 'No'}")
    
    if result["damage_assessment"]:
        print(f"\nDAMAGE ASSESSMENT:")
        print(f"Damage Level: {result['damage_assessment'].get('damage_level', 'N/A')}")
        print(f"Description: {result['damage_assessment'].get('damage_description', 'N/A')}")
        print(f"Estimated Repair Cost: ${result['damage_assessment'].get('estimated_repair_cost', 0):.2f}")
        print(f"Recommendation: {result['damage_assessment'].get('recommendation', 'N/A')}")
    
    if result["payment_amount"] > 0:
        print(f"\nPAYMENT INFORMATION:")
        print(f"Payment Amount: ${result['payment_amount']:.2f}")
        print(f"Payment Authorized: {'Yes' if result['payment_authorized'] else 'No'}")
    
    print(f"\nMESSAGE LOG:")
    for i, msg in enumerate(result["messages"], 1):
        print(f"{i}. {msg}")
    
    print(border + "\n")


def run_test_scenarios():
    """
    Run through all the test scenarios required in the assignment.
    """
    print("Running test scenarios for Allstate Auto Insurance Claim Settlement System...\n")
    
    # Initialize placeholder variables
    major_damage_image = "placeholder_image_base64_data"
    minor_damage_image = "placeholder_image_base64_data"
    moderate_damage_image = "placeholder_image_base64_data"
    
    # Try to load real damage images
    try:
        major_damage_image = load_image_base64("test_images/car_damage_major.jpg")
        minor_damage_image = load_image_base64("test_images/car_damage_minor.jpg")
        moderate_damage_image = load_image_base64("test_images/car_damage_moderate.jpg")
        
        print("Successfully loaded damage images from test_images folder")
    except Exception as e:
        print(f"Warning: Could not load image files: {str(e)}")
        print("Using test mode with simulated damage assessments instead")
    
   # Scenario 1: Delinquent insured
    print("\n" + "="*80)
    print("SCENARIO 1: POLICY VERIFICATION AGENT RECOGNIZES A DELINQUENT INSURED")
    print("="*80 + "\n")
    
    # For testing, we'll use a policy number ending in 0 to simulate a delinquent policy
    delinquent_policy = "999 876 540"
    # Force test mode to ensure predictable results
    result_1 = process_claim(delinquent_policy, minor_damage_image, test_mode=False, test_damage_level="minor")
    display_claim_results(result_1, "Delinquent Insured")
    
    # Scenario 2: Invalid policy number
    print("\n" + "="*80)
    print("SCENARIO 2: POLICY VERIFICATION AGENT RECOGNIZES AN INVALID POLICY NUMBER")
    print("="*80 + "\n")
    
    invalid_policy = "111 222 333"  # Not in our valid policies list
    result_2 = process_claim(invalid_policy, minor_damage_image, test_mode=False)
    display_claim_results(result_2, "Invalid Policy Number")
    
    # Scenario 3: Major damage
    print("\n" + "="*80)
    print("SCENARIO 3: ASSESSMENT AGENT RECOGNIZES MAJOR DAMAGE")
    print("="*80 + "\n")
    
    valid_policy = "999 876 543"
    # Use the actual major damage image if possible, fallback to test mode if needed
    if major_damage_image != "placeholder_image_base64_data":
        result_3 = process_claim(valid_policy, major_damage_image)
    else:
        result_3 = process_claim(valid_policy, major_damage_image, test_mode=False, test_damage_level="major")
    display_claim_results(result_3, "Major Damage")
    
    # Scenario 4: Minor damage
    print("\n" + "="*80)
    print("SCENARIO 4: ASSESSMENT AGENT RECOGNIZES MINOR DAMAGE")
    print("="*80 + "\n")
    
    # Use the actual minor damage image if possible, fallback to test mode if needed
    if minor_damage_image != "placeholder_image_base64_data":
        result_4 = process_claim(valid_policy, minor_damage_image)
    else:
        result_4 = process_claim(valid_policy, minor_damage_image, test_mode=False, test_damage_level="minor")
    display_claim_results(result_4, "Minor Damage")
    
    # Scenario 5: Moderate damage
    print("\n" + "="*80)
    print("SCENARIO 5: ASSESSMENT AGENT RECOGNIZES MODERATE DAMAGE")
    print("="*80 + "\n")
    
    # Use the actual moderate damage image if possible, fallback to test mode if needed
    if moderate_damage_image != "placeholder_image_base64_data":
        result_5 = process_claim(valid_policy, moderate_damage_image)
    else:
        result_5 = process_claim(valid_policy, moderate_damage_image, test_mode=False, test_damage_level="moderate")
    display_claim_results(result_5, "Moderate Damage")
    
    print("\nAll test scenarios completed.")


def run_with_real_data(policy_number: str, image_path: str):
    """
    Run the system with actual data.
    
    Args:
        policy_number: The actual policy number
        image_path: Path to the actual image of damage
    """
    try:
        # Load the image file
        damage_image_base64 = load_image_base64(image_path)
        
        # Process the claim - note we're NOT using test_mode here
        result = process_claim(policy_number, damage_image_base64)
        
        # Display results
        display_claim_results(result, "Real Data Test")
        
        return result
    except Exception as e:
        print(f"Error processing claim with real data: {str(e)}")
        import traceback
        traceback.print_exc()  # This will show the full error stack
        return None


# Main function to run the program
def main():
    # Run all test scenarios
    run_test_scenarios()
    
    # Uncomment to run with real data:
    # run_with_real_data("999 876 543", "test_images/car_damage_major.jpg")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()