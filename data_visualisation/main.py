import subprocess
import json
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool

from data_visualisation.prompts import orchestrator_agent_prompt, data_agent_prompt
from data_visualisation.graphs import create_line_graph, create_bar_chart, create_pie_chart, create_scatter_plot


llm = ChatOllama(
    model="qwen3:32b",
    base_url="http://localhost:11434",
)

DATA_PATH = "/home/robust/voice-demo-new/data_visualisation/playground/sales_data_sample.csv"
PYTHON_OUTPUT_PATH="/home/robust/voice-demo-new/data_visualisation/playground/extract_data_response.py"

@tool
def extract_data(query: str) -> str:
    """Extract and aggregate data from the sales dataset based on a natural language query.
    
    Args:
        query: The user's EXACT original query - pass it verbatim without rephrasing.
            IMPORTANT: Do NOT modify, rephrase, or reinterpret the user's request.
            Pass the user's words exactly as they said them.
            
            If user says "Show me total sales by year for 2005", pass exactly that.
            Do NOT change years, metrics, groupings, or any other details.
    
    Returns:
        Structured data (dict) containing the extracted/aggregated results, or an error message if execution fails.
    """
    print("Orchestrator Query:", query)
    response = llm.invoke([
        (
            "system",
            data_agent_prompt.format(data_file=DATA_PATH)
        ),
        (
            "human",
            query
        )
    ])

    with open(PYTHON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(response.content)

    # Run the generated Python code and capture the output
    result = subprocess.run(
        # [".venv/Scripts/python", PYTHON_OUTPUT_PATH],
        ["/home/robust/voice-demo-new/voice-sandwich-demo/components/python/.venv/bin/python", PYTHON_OUTPUT_PATH],
        
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        chart_data = json.loads(result.stdout)
        print("Chart Output:", chart_data)
        return chart_data
    else:
        print("Error:", result.stderr)
        return f"Error executing generated code: {result.stderr}"

@tool
def create_chart(graph_type, x_data, y_data, title, x_label, y_label):
    """Generate a chart image based on the provided data and configuration.
    
    Args:
        graph_type: The type of chart to create:
            - "line" for line graphs
            - "bar" for bar charts
            - "pie" for pie charts
            - "scatter" for scatter plots
        x_data: List of values for the x-axis (e.g., dates, categories, labels).
        y_data: List of numerical values for the y-axis, aligned with x_data.
        title: The title to display on the chart.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
    
    Returns:
        The file path where the chart image was saved.
    """
    if graph_type == "line":
        return create_line_graph(
            x_data=x_data,
            y_data=y_data,
            title=title,
            x_label=x_label,
            y_label=y_label
        )
    elif graph_type == "bar":
        return create_bar_chart(
            x_data=x_data,
            y_data=y_data,
            title=title,
            x_label=x_label,
            y_label=y_label
        )
    elif graph_type == "pie":
        return create_pie_chart(
            labels=x_data,
            values=y_data,
            title=title
        )
    elif graph_type == "scatter":
        return create_scatter_plot(
            x_data=x_data,
            y_data=y_data,
            title=title,
            x_label=x_label,
            y_label=y_label
        )
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")


def test():
    print("Testing data extraction...")

from langchain_groq import ChatGroq
llm = ChatGroq(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        groq_api_key="",
        model="openai/gpt-oss-20b",
        temperature=0.3,
        # max_tokens=max_tokens,
    )

def main(llm, query):
    tools = [extract_data, create_chart]    
    orchestrator_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=orchestrator_agent_prompt,
        name="orchestrator_agent",
    )

    response = orchestrator_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return response

def main2(llm):
    tools = [extract_data, create_chart]    
    orchestrator_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=orchestrator_agent_prompt,
        name="orchestrator_agent",
    )

    # response = orchestrator_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return orchestrator_agent


if __name__ == "__main__":
    questions = [
        # Sales Performance
        "Show me total sales by year",
        "What are the monthly sales trends for 2004?",
        "Show quarterly sales comparison across all years",

        # Product Analysis
        "Which product line generates the most revenue?",
        "Show me average order value by product line",
        "What are the top 10 best-selling products by quantity?",

        # Geographic Insights
        "Show total sales by country",
        "Which cities have the highest sales?",
        "Compare sales across territories",

        # Customer Analysis
        "Who are the top 10 customers by total sales?",
        "Show me customer count by country",
        "What is the average order size by customer?",

        # Deal & Order Analysis
        "Show sales distribution by deal size",
        "What is the order status breakdown?",
        "Show cancelled orders by product line",

        # Time-Based Analysis
        "What were the best performing months overall?",
        "Show daily sales for December 2004",
        "Compare Q4 sales across all years",

        # Pricing Analysis
        "Show average price vs MSRP by product line",
        "What is the relationship between quantity ordered and deal size?"
    ]

    query = questions[2]
    print("User Query:", query)
    result = main(llm, query)
    print("Orchestrator Agent Response:\n", result)
