import os
import base64
from typing import Dict, TypedDict, Annotated, List, Literal, Any
from enum import Enum
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

# Setup OpenAI API key
os.environ["OPENAI_API_KEY"] = "key here"

# Models
gpt4_vision = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4096)
gpt4 = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=2048)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Define the possible claim status values
class ClaimStatus(str, Enum):
    SUBMITTED = "submitted"
    PENDING_VERIFICATION = "pending_verification"
    POLICY_VERIFIED = "policy_verified"
    DENIED_INVALID_POLICY = "denied_invalid_policy"
    DENIED_DELINQUENT = "denied_delinquent"
    DAMAGE_ASSESSMENT_PENDING = "damage_assessment_pending"
    DAMAGE_ASSESSED = "damage_assessed" 
    PAYMENT_PROCESSING = "payment_processing"
    PAYMENT_COMPLETE = "payment_complete"
    CLAIM_APPROVED = "claim_approved"
    CLAIM_DENIED = "claim_denied"
    COMPLETED = "completed"

# Define the State Type that will be passed between nodes
class ClaimState(TypedDict):
    # Customer information
    policy_number: str
    damage_image: str
    
    # Policy verification information
    policy_verified: bool
    policy_verification_result: str
    is_delinquent: bool
    
    # Damage assessment information
    damage_assessment: Dict
    damage_report_generated: bool
    
    # Claim processing information
    policy_holder_records_updated: bool
    
    # Payment information
    payment_amount: float
    payment_processed: bool
    
    # Overall claim status
    claim_status: str
    
    # Communication
    notifications: List[Dict]
    messages: List[str]

# --------- POLICY DATA STORE ---------

class PolicyDocument:
    def __init__(self, policy_number: str, vehicle_info: str, policy_holder: str, 
                 effective_date: str, payment_status: str = "current"):
        self.policy_number = policy_number
        self.vehicle_info = vehicle_info
        self.policy_holder = policy_holder
        self.effective_date = effective_date
        self.payment_status = payment_status  # "current" or "delinquent"
    
    def to_document(self) -> Document:
        """Convert to a Document object for vectorstore"""
        content = f"""
        Policy Number: {self.policy_number}
        Vehicle: {self.vehicle_info}
        Policy Holder: {self.policy_holder}
        Effective Date: {self.effective_date}
        Payment Status: {self.payment_status}
        """
        metadata = {
            "policy_number": self.policy_number,
            "vehicle_info": self.vehicle_info,
            "policy_holder": self.policy_holder,
            "effective_date": self.effective_date,
            "payment_status": self.payment_status
        }
        return Document(page_content=content, metadata=metadata)

class PolicyDataStore:
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        self.policies = []
        self.vectorstore = None
        
    def add_policy(self, policy: PolicyDocument):
        """Add a policy to the data store"""
        self.policies.append(policy)
    
    def build_vectorstore(self):
        """Build the vectorstore from all policies"""
        documents = [policy.to_document() for policy in self.policies]
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
    
    def verify_policy(self, policy_number: str) -> Dict:
        """Verify if a policy exists and is valid"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not built. Call build_vectorstore() first.")
        
        query = f"Policy Number: {policy_number}"
        results = self.vectorstore.similarity_search(query, k=1)
        
        if not results:
            return {
                "policy_valid": False,
                "delinquent": False,
                "verification_message": f"Policy verification failed. The policy number {policy_number} is not found in our records."
            }
        
        # Get the best match
        best_match = results[0]
        metadata = best_match.metadata
        
        # Check if the policy number matches exactly
        if metadata["policy_number"].strip() == policy_number.strip():
            is_delinquent = metadata["payment_status"].lower() == "delinquent"
            
            if is_delinquent:
                return {
                    "policy_valid": True,
                    "delinquent": True,
                    "verification_message": f"Policy {policy_number} is valid but payment is delinquent. Claim is denied until payments are brought current."
                }
            else:
                return {
                    "policy_valid": True,
                    "delinquent": False,
                    "verification_message": f"Policy {policy_number} verified successfully. The policy is active and payments are up to date."
                }
        else:
            # Not an exact match
            return {
                "policy_valid": False,
                "delinquent": False,
                "verification_message": f"Policy verification failed. The policy number {policy_number} is not found in our records."
            }

# Initialize and populate the policy data store
policy_store = PolicyDataStore(embeddings_model)

# Add sample policies to the data store
policy_store.add_policy(PolicyDocument(
    policy_number="999 876 543",
    vehicle_info="Honda Accord 2022",
    policy_holder="John Smith",
    effective_date="January 14, 2025"
))

policy_store.add_policy(PolicyDocument(
    policy_number="123 456 789",
    vehicle_info="Toyota Camry 2023",
    policy_holder="Jane Doe",
    effective_date="February 20, 2025"
))

policy_store.add_policy(PolicyDocument(
    policy_number="987 654 321",
    vehicle_info="Ford F-150 2021",
    policy_holder="Robert Johnson",
    effective_date="March 5, 2025"
))

policy_store.add_policy(PolicyDocument(
    policy_number="999 876 540",
    vehicle_info="Chevrolet Malibu 2022",
    policy_holder="Michael Wilson",
    effective_date="December 10, 2024",
    payment_status="delinquent"
))

# Build the vectorstore
policy_store.build_vectorstore()

# --------- DAMAGE DATA STORE ---------

class DamageReference:
    def __init__(self, 
                 damage_level: str, 
                 damage_description: str, 
                 estimated_repair_cost: float,
                 recommendation: str):
        self.damage_level = damage_level
        self.damage_description = damage_description
        self.estimated_repair_cost = estimated_repair_cost
        self.recommendation = recommendation
    
    def to_document(self) -> Document:
        """Convert to a Document object for vectorstore"""
        content = f"""
        Damage Level: {self.damage_level}
        Description: {self.damage_description}
        Estimated Repair Cost: ${self.estimated_repair_cost:.2f}
        Recommendation: {self.recommendation}
        """
        metadata = {
            "damage_level": self.damage_level,
            "damage_description": self.damage_description,
            "estimated_repair_cost": self.estimated_repair_cost,
            "recommendation": self.recommendation
        }
        return Document(page_content=content, metadata=metadata)

class DamageDataStore:
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        self.damage_references = []
        self.vectorstore = None
        
    def add_damage_reference(self, damage_ref: DamageReference):
        """Add a damage reference to the data store"""
        self.damage_references.append(damage_ref)
    
    def build_vectorstore(self):
        """Build the vectorstore from all damage references"""
        documents = [ref.to_document() for ref in self.damage_references]
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
    
    def get_combined_assessment(self, damage_description: str) -> Dict:
        """Get a combined assessment based on similar damage references"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not built. Call build_vectorstore() first.")
            
        similar_damages = self.vectorstore.similarity_search(damage_description, k=3)
        
        if not similar_damages:
            # Fallback assessment
            return {
                "damage_level": "moderate",
                "damage_description": damage_description,
                "estimated_repair_cost": 1500.00,
                "recommendation": "Approve claim for standard repair procedure"
            }
        
        # Calculate assessment based on similarity
        damage_levels = {"minor": 0, "moderate": 0, "major": 0}
        total_cost = 0.0
        recommendations = []
        
        for doc in similar_damages:
            metadata = doc.metadata
            damage_levels[metadata["damage_level"]] += 1
            total_cost += float(metadata["estimated_repair_cost"])
            recommendations.append(metadata["recommendation"])
        
        # Determine most frequent damage level
        damage_level = max(damage_levels, key=damage_levels.get)
        
        # Average cost
        avg_cost = total_cost / len(similar_damages)
        
        # Pick a recommendation based on damage level
        if damage_level == "minor":
            recommendation = "Approve claim for minor cosmetic repair."
        elif damage_level == "moderate":
            recommendation = "Approve claim for standard repair procedure."
        else:  # major
            recommendation = "Approve claim for extensive repair. Possible frame damage should be inspected."
        
        return {
            "damage_level": damage_level,
            "damage_description": damage_description,
            "estimated_repair_cost": round(avg_cost, 2),
            "recommendation": recommendation
        }

# Initialize the damage data store
damage_store = DamageDataStore(embeddings_model)

# Add sample damage references
damage_store.add_damage_reference(DamageReference(
    damage_level="minor",
    damage_description="Light scratches and small dent on the front bumper",
    estimated_repair_cost=750.00,
    recommendation="Approve claim for minor cosmetic repair."
))

damage_store.add_damage_reference(DamageReference(
    damage_level="moderate",
    damage_description="Damaged front bumper with dents and broken headlight",
    estimated_repair_cost=2200.00,
    recommendation="Approve claim for standard repair procedure."
))

damage_store.add_damage_reference(DamageReference(
    damage_level="major",
    damage_description="Severe front-end damage with bumper detachment, broken headlights, and hood damage",
    estimated_repair_cost=5800.00,
    recommendation="Approve claim for extensive repair. Possible frame damage should be inspected."
))

# Build the damage vectorstore
damage_store.build_vectorstore()

# --------- CUSTOMER AGENT TOOLS ---------

@tool
def submit_claim(policy_number: str, damage_image: str) -> Dict:
    """
    Tool for a customer to submit an insurance claim
    
    Args:
        policy_number: The policy number of the insured
        damage_image: Base64 encoded image of the car damage
        
    Returns:
        A dictionary with the initial claim information
    """
    return {
        "claim_id": f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "policy_number": policy_number,
        "submission_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "status": "submitted",
        "message": f"Claim for policy {policy_number} has been submitted successfully and sent to Policy Verification."
    }

# --------- POLICY VERIFICATION AGENT TOOLS ---------

@tool
def verify_policy(policy_number: str) -> Dict:
    """
    Verifies if the policy number is valid and if the insured is not delinquent
    
    Args:
        policy_number: The policy number to verify
        
    Returns:
        A dictionary with verification results
    """
    return policy_store.verify_policy(policy_number)

@tool
def check_delinquency(policy_number: str) -> Dict:
    """
    Checks if the policy holder is delinquent on payments
    
    Args:
        policy_number: The policy number to check
        
    Returns:
        A dictionary with delinquency check results
    """
    verification_result = policy_store.verify_policy(policy_number)
    
    if not verification_result["policy_valid"]:
        return {
            "delinquent": False,
            "message": f"Policy {policy_number} not found. Cannot check delinquency status."
        }
    
    if verification_result["delinquent"]:
        return {
            "delinquent": True,
            "message": "The policy holder is delinquent on payments. Claim processing is denied."
        }
    else:
        return {
            "delinquent": False,
            "message": "The policy holder is current on all payments."
        }

# --------- DAMAGE ASSESSMENT AGENT TOOLS ---------

@tool
def assess_damage(damage_image: str) -> Dict:
    """
    Assesses the damage shown in an image using AI vision capabilities
    
    Args:
        damage_image: Base64 encoded image of the car damage
        
    Returns:
        A dictionary with damage assessment results
    """
    # Use GPT-4 Vision to analyze the image
    damage_prompt = PromptTemplate.from_template(
        """You are an expert auto insurance damage assessor. Analyze the image of car damage and provide:
        1. A detailed description of the visible damage
        2. Classify the damage as minor, moderate, or major
        
        Be very specific and detailed about the damage you observe.
        """
    )
    
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": damage_prompt.format(image="")},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{damage_image}"}}
            ]
        )
    ]
    
    # Get description from vision model
    response = gpt4_vision.invoke(messages)
    damage_description = response.content
    
    # Use damage data store to get assessment
    assessment_result = damage_store.get_combined_assessment(damage_description)
    assessment_result["damage_description"] = damage_description
    
    return assessment_result

@tool
def generate_damage_report(damage_assessment: Dict) -> Dict:
    """
    Generates a formal damage report based on the damage assessment
    
    Args:
        damage_assessment: The damage assessment data
        
    Returns:
        A dictionary with the damage report information
    """
    damage_level = damage_assessment.get("damage_level", "moderate")
    damage_description = damage_assessment.get("damage_description", "Unspecified damage")
    repair_cost = damage_assessment.get("estimated_repair_cost", 1500.00)
    
    report = {
        "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "damage_level": damage_level,
        "detailed_description": damage_description,
        "estimated_repair_cost": repair_cost,
        "recommended_action": damage_assessment.get("recommendation", "Standard repair procedure recommended"),
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return report

# --------- CLAIM PROCESSING AGENT TOOLS ---------

@tool
def update_policy_holder_records(policy_number: str, damage_report: Dict) -> Dict:
    """
    Updates the policy holder's records with the claim and damage information
    
    Args:
        policy_number: The policy number
        damage_report: The damage report
        
    Returns:
        A dictionary with the update results
    """
    return {
        "policy_number": policy_number,
        "records_updated": True,
        "update_type": "Claim and Damage Information",
        "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "message": f"Policy holder records for {policy_number} updated with damage report {damage_report['report_id']}"
    }

@tool
def send_notification(recipient: str, claim_status: str, message: str) -> Dict:
    """
    Sends a notification to the insured person
    
    Args:
        recipient: The policy number of the recipient
        claim_status: The current status of the claim
        message: The notification message
        
    Returns:
        A dictionary with notification results
    """
    return {
        "sent": True,
        "recipient": recipient,
        "claim_status": claim_status,
        "message": message,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# --------- CLAIM PAYMENT AGENT TOOLS ---------

@tool
def process_claim_payment(policy_number: str, damage_report: Dict) -> Dict:
    """
    Processes the payment for the claim
    
    Args:
        policy_number: The policy number
        damage_report: The damage report with cost information
        
    Returns:
        A dictionary with payment processing results
    """
    damage_level = damage_report.get("damage_level", "moderate")
    estimated_cost = damage_report.get("estimated_repair_cost", 1500.00)
    
    # Apply coverage and deductible based on damage level
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
        "payment_id": f"PAY-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "policy_number": policy_number,
        "payment_amount": round(payment_amount, 2),
        "deductible_applied": deductible,
        "coverage_percentage": coverage_percentage,
        "payment_date": datetime.now().strftime('%Y-%m-%d'),
        "status": "processed",
        "message": f"Payment of ${payment_amount:.2f} processed for claim on policy {policy_number}"
    }

# --------- AGENT FUNCTIONS ---------

def customer_agent(state: ClaimState) -> ClaimState:
    """Agent representing the customer who submits the claim"""
    # If this is a new claim submission
    if state["claim_status"] == ClaimStatus.SUBMITTED:
        claim_result = submit_claim.invoke({
            "policy_number": state["policy_number"],
            "damage_image": state["damage_image"]
        })
        
        # Update state with submission information
        state["messages"].append(claim_result["message"])
        state["claim_status"] = ClaimStatus.PENDING_VERIFICATION
    
    return state

def policy_verification_agent(state: ClaimState) -> ClaimState:
    """Agent that verifies the policy and checks for delinquency"""
    # Only process if we're in the pending verification state
    if state["claim_status"] != ClaimStatus.PENDING_VERIFICATION:
        return state
        
    policy_number = state["policy_number"]
    
    # Step 1: Verify if the policy is valid
    verification_result = verify_policy.invoke({"policy_number": policy_number})
    policy_verified = verification_result["policy_valid"]
    
    # Update state with verification results
    state["policy_verified"] = policy_verified
    state["policy_verification_result"] = verification_result["verification_message"]
    state["messages"].append(verification_result["verification_message"])
    
    # Step 2: Check for delinquency if policy is valid
    if policy_verified:
        delinquency_result = check_delinquency.invoke({"policy_number": policy_number})
        state["is_delinquent"] = delinquency_result["delinquent"]
        state["messages"].append(delinquency_result["message"])
        
        # Handle delinquent policies
        if delinquency_result["delinquent"]:
            state["claim_status"] = ClaimStatus.DENIED_DELINQUENT
            
            # Send denial notification
            notification = send_notification.invoke({
                "recipient": policy_number,
                "claim_status": str(ClaimStatus.DENIED_DELINQUENT),
                "message": "Your claim has been denied because your policy is delinquent. Please contact customer service."
            })
            
            state["notifications"].append(notification)
            state["messages"].append(f"Notification sent: {notification['message']}")
        else:
            # Policy is valid and not delinquent, proceed to damage assessment
            state["claim_status"] = ClaimStatus.DAMAGE_ASSESSMENT_PENDING
    else:
        # Invalid policy
        state["claim_status"] = ClaimStatus.DENIED_INVALID_POLICY
        
        # Send denial notification
        notification = send_notification.invoke({
            "recipient": policy_number,
            "claim_status": str(ClaimStatus.DENIED_INVALID_POLICY),
            "message": "Your claim has been denied because the policy number is invalid. Please check and resubmit."
        })
        
        state["notifications"].append(notification)
        state["messages"].append(f"Notification sent: {notification['message']}")
    
    return state

def damage_assessment_agent(state: ClaimState) -> ClaimState:
    """Agent that assesses the damage based on the image"""
    # Only process if we're in the damage assessment pending state
    if state["claim_status"] != ClaimStatus.DAMAGE_ASSESSMENT_PENDING:
        return state
    
    # Step 1: Assess the damage from the image
    damage_assessment_result = assess_damage.invoke({
        "damage_image": state["damage_image"]
    })
    
    # Update state with assessment results
    state["damage_assessment"] = damage_assessment_result
    state["messages"].append(f"Damage assessment: {damage_assessment_result['damage_level']} damage. {damage_assessment_result['damage_description']}")
    
    # Step 2: Generate a formal damage report
    damage_report = generate_damage_report.invoke({
        "damage_assessment": damage_assessment_result
    })
    
    # Add the report to the damage assessment
    state["damage_assessment"]["report"] = damage_report
    state["damage_report_generated"] = True
    state["messages"].append(f"Damage report generated: {damage_report['report_id']}")
    
    # Update state to indicate damage has been assessed
    state["claim_status"] = ClaimStatus.DAMAGE_ASSESSED
    
    return state

def claim_processing_agent(state: ClaimState) -> ClaimState:
    """Agent that processes the claim and updates records"""
    # Only process if damage has been assessed
    if state["claim_status"] != ClaimStatus.DAMAGE_ASSESSED:
        return state
    
    # Update policy holder records with claim and damage information
    update_result = update_policy_holder_records.invoke({
        "policy_number": state["policy_number"],
        "damage_report": state["damage_assessment"]["report"]
    })
    
    # Update state
    state["policy_holder_records_updated"] = update_result["records_updated"]
    state["messages"].append(update_result["message"])
    
    # Move to payment processing
    state["claim_status"] = ClaimStatus.PAYMENT_PROCESSING
    
    return state

def claim_payment_agent(state: ClaimState) -> ClaimState:
    """Agent that handles payment for the claim"""
    # Only process if we're in payment processing state
    if state["claim_status"] != ClaimStatus.PAYMENT_PROCESSING:
        return state
    
    # Process the payment
    payment_result = process_claim_payment.invoke({
        "policy_number": state["policy_number"],
        "damage_report": state["damage_assessment"]["report"]
    })
    
    # Update state with payment information
    state["payment_amount"] = payment_result["payment_amount"]
    state["payment_processed"] = True
    state["messages"].append(payment_result["message"])
    
    # Move to approved state
    state["claim_status"] = ClaimStatus.CLAIM_APPROVED
    
    # Send approval notification to customer
    notification = send_notification.invoke({
        "recipient": state["policy_number"],
        "claim_status": str(ClaimStatus.CLAIM_APPROVED),
        "message": f"Your claim has been approved. Payment of ${payment_result['payment_amount']:.2f} has been processed."
    })
    
    state["notifications"].append(notification)
    state["messages"].append(f"Notification sent: {notification['message']}")
    
    # Mark as completed
    state["claim_status"] = ClaimStatus.COMPLETED
    
    return state

# --------- CONDITIONAL ROUTING FUNCTIONS ---------

def route_after_customer_submission(state: ClaimState) -> Literal["policy_verification", "end"]:
    """Routes the flow after customer submits a claim"""
    if state["claim_status"] == ClaimStatus.PENDING_VERIFICATION:
        return "policy_verification"
    return "end"

def route_after_policy_verification(state: ClaimState) -> Literal["damage_assessment", "end"]:
    """Routes the flow after policy verification"""
    if state["claim_status"] == ClaimStatus.DAMAGE_ASSESSMENT_PENDING:
        return "damage_assessment"
    return "end"

def route_after_damage_assessment(state: ClaimState) -> Literal["claim_processing", "end"]:
    """Routes the flow after damage assessment"""
    if state["claim_status"] == ClaimStatus.DAMAGE_ASSESSED:
        return "claim_processing"
    return "end"

def route_after_claim_processing(state: ClaimState) -> Literal["claim_payment", "end"]:
    """Routes the flow after claim processing"""
    if state["claim_status"] == ClaimStatus.PAYMENT_PROCESSING:
        return "claim_payment"
    return "end"

# --------- BUILD THE GRAPH ---------

def build_insurance_claim_graph():
    """Builds the graph for insurance claim processing according to the BPMN diagram"""
    # Initialize the graph
    workflow = StateGraph(ClaimState)
    
    # Add nodes for each agent/role in the BPMN diagram
    workflow.add_node("customer", customer_agent)
    workflow.add_node("policy_verification", policy_verification_agent)
    workflow.add_node("damage_assessment", damage_assessment_agent)
    workflow.add_node("claim_processing", claim_processing_agent)
    workflow.add_node("claim_payment", claim_payment_agent)
    
    # Add conditional edges following the BPMN flow
    workflow.add_conditional_edges(
        "customer",
        route_after_customer_submission,
        {
            "policy_verification": "policy_verification",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "policy_verification",
        route_after_policy_verification,
        {
            "damage_assessment": "damage_assessment",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "damage_assessment",
        route_after_damage_assessment,
        {
            "claim_processing": "claim_processing",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "claim_processing",
        route_after_claim_processing,
        {
            "claim_payment": "claim_payment",
            "end": END
        }
    )
    
    # Add final edge
    workflow.add_edge("claim_payment", END)
    
    # Set the entry point
    workflow.set_entry_point("customer")
    
    return workflow.compile()

# --------- PROCESS A CLAIM ---------

def process_claim(policy_number: str, damage_image_base64: str):
    """
    Processes a claim through the entire workflow
    
    Args:
        policy_number: The policy number for the claim
        damage_image_base64: Base64 encoded image of the damage
    
    Returns:
        The final state after processing the claim
    """
    # Build the graph
    graph = build_insurance_claim_graph()
    
    # Initialize the state
    initial_state = ClaimState(
        policy_number=policy_number,
        damage_image=damage_image_base64,
        policy_verified=False,
        policy_verification_result="",
        is_delinquent=False,
        damage_assessment={},
        damage_report_generated=False,
        policy_holder_records_updated=False,
        payment_amount=0.0,
        payment_processed=False,
        claim_status=ClaimStatus.SUBMITTED,
        notifications=[],
        messages=[]
    )
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result

# Example usage
if __name__ == "__main__":
    # Load test image
    def load_image_base64(image_path: str) -> str:
        """Loads an image and converts it to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
            
    # You would replace this with your actual image path
    try:
        damage_image = load_image_base64("test_images/car_damage_moderate.jpg")
    except:
        # Placeholder for demo
        damage_image = "placeholder_image_base64_data"
    
    # Process a claim
    valid_policy = "999 876 543"
    result = process_claim(valid_policy, damage_image)
    
    # Display results
    print(f"CLAIM STATUS: {result['claim_status']}")
    print(f"MESSAGES:")
    for msg in result["messages"]:
        print(f"- {msg}")