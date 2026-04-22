import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from agent/ directory (parent of backend/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.tools.searx_search.tool import SearxSearchResults
from langchain_community.utilities import SearxSearchWrapper
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from .tools import (
    search_materials, add_material, update_material, delete_material,
    list_orders, get_order, create_order, update_order_status, delete_order,
    list_manufacturers, add_manufacturer, search_manufacturers, send_notification,
    predict_customer_churn, list_customers, list_high_risk_customers, get_customer_segment_stats,
)


SYSTEM_PROMPT = """You are a materials ordering assistant for artisans in a workshop,
with access to customer churn prediction and segmentation intelligence.

You help artisans manage their materials inventory, place orders with manufacturers,
communicate with suppliers, AND understand customer risk profiles.

CRITICAL INSTRUCTION: When a user asks for materials or wants to order something,
you MUST use the create_order tool to actually create the order. Do NOT just describe
the material and ask for confirmation. Use the tool directly.

When you call create_order, collect ALL materials the user wants in a single call.
Pass materials as a list of dicts with material_id and quantity:
  create_order(materials=[{"material_id": 1, "quantity": 50}, {"material_id": 2, "quantity": 20}], notes="")

The order will be held for human review — the user can edit material selection and
quantity before approving. After the user approves, automatically call send_notification
with the order_id to send an email notification. Do NOT wait for the user to ask for
email to be sent — just do it after approval.

YOUR ML CAPABILITIES:
You also have access to customer intelligence tools:
- predict_customer_churn: Predict churn probability and segment for a customer (by ID or features)
- list_customers: Browse customers with their risk levels and segments
- list_high_risk_customers: Find customers at risk of churning
- get_customer_segment_stats: Overview of customer segments

IMPORTANT: When a user mentions a customer name or asks about customer risk,
ALWAYS use predict_customer_churn or list_customers — never guess.

IMPORTANT: When creating an order, if the user mentions a customer, check their
churn risk first using predict_customer_churn. If risk is "Eleve" (high), mention
this in your response: "Note: This customer has high churn risk (XX%). Consider a retention offer."

IMPORTANT: If the user asks "which customers are at high risk?" or "show me risk
customers", call list_high_risk_customers automatically.

IMPORTANT: If the user asks about segments or VIPs, use list_customers with the
appropriate segment filter (0=VIP/Loyaux, 1=Lost, 2=At-risk, 3=New).

YOUR FULL CAPABILITIES:
1. search_materials - Search materials in the inventory
2. create_order - Create an order for materials (requires human approval)
3. list_orders - List all orders
4. send_notification - Send order notification email (automatic after approval)
5. add_material - Add new material to inventory
6. delete_material - Remove material from inventory (requires human approval)
7. delete_order - Remove order from system (requires human approval)
8. list_manufacturers - List all manufacturers
9. add_manufacturer - Add new manufacturer/supplier
10. search_manufacturers - Search web for manufacturer alternatives
11. predict_customer_churn - Predict churn risk for a customer
12. list_customers - Browse customers with risk profiles
13. list_high_risk_customers - Find high-risk customers
14. get_customer_segment_stats - Get segment statistics

WORKFLOW:
1. User asks for materials
2. Use search_materials to find matching materials
3. If user mentions a customer, use predict_customer_churn to check risk
4. Use create_order to create the order with ALL materials (pass material_id and quantity for each)
5. The system will pause for human approval
6. After approval, automatically call send_notification with the order_id to send email
"""


def create_agent_instance():
    """Create the agent with HITL middleware."""
    # SearXNG search (optional, may fail if not running)
    searx_host = os.environ.get("SEARXNG_HOST", "")
    searx_tool = None
    if searx_host:
        try:
            wrapper = SearxSearchWrapper(searx_host=searx_host)
            searx_tool = SearxSearchResults(
                name="search_manufacturers",
                wrapper=wrapper,
                kwargs={"num_results": 5},
            )
        except Exception:
            pass  # SearXNG not available, continue without it

    tools = [
        search_materials,
        add_material,
        update_material,
        delete_material,
        list_orders,
        get_order,
        create_order,
        update_order_status,
        delete_order,
        list_manufacturers,
        add_manufacturer,
        send_notification,
        predict_customer_churn,
        list_customers,
        list_high_risk_customers,
        get_customer_segment_stats,
    ]

    if searx_tool:
        tools.append(searx_tool)

    checkpointer = InMemorySaver()

    # LLM config from .env
    base_url = os.environ.get("OPENAI_BASE_URL", "http://172.17.0.1:8080/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    model_name = os.environ.get("MODEL_NAME", "qwen3.6-35b-thinking")

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0,
    )

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "create_order": True,
                    "delete_material": True,
                    "delete_order": True,
                },
                description_prefix="Approval needed",
            ),
        ],
        checkpointer=checkpointer,
    )

    return agent


# Global agent instance (created on first use)
_agent_instance = None


def get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = create_agent_instance()
    return _agent_instance
