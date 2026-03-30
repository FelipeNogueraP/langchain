from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable


load_dotenv()

MAX_ITERATIONS = 3
MODEL = "qwen3:1.7b"

# Tools

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product."""
    print(f"Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount according to the discount tier."""
    print(f" Executing applying_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold":23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount /100), 2)


# Agent loop

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
    ),

    HumanMessage(content=question),
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {i}")
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)
        tool_calls = ai_message.tool_calls

        # If not tool calls, we assume the model is done and has generated a final answer.
        if not tool_calls:
            print("No tool calls detected. Final answer:")
            print(ai_message.content)
            return ai_message.content
        
        # Process only the first tool.
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args") or tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool selected] {tool_name}")

        tool_to_use = tools_dict.get(tool_name)
        if not tool_to_use:
            print(f"Error: Tool '{tool_name}' not found.")
            return "Error: Invalid tool selected by the agent."
        
        observation = tool_to_use.invoke(tool_args)
        print(f"[Observation] {observation}")

        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    return "Error: Reached the maximum number of iterations without a final answer."

if __name__ == "__main__":
    print("Agent with tools running.")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(result)
