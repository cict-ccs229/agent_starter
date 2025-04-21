from smolagents import CodeAgent, HfApiModel, tool
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI
import os
from supabase import create_client, Client
import yaml
import pandas as pd

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

@tool
def select(query_snippet: str) -> str:
    """A tool that selects data from the medicinal_plants table.
    
    Args:
        query_snippet (str): The query snippet to execute. Can be reserved sql keywords or field names. Does not include the table name, conditions or filters.
    
    Returns:
        str: The result of the query as a markdown table.
    """
    
    try:
        response = supabase.table("medicinal_plants").select(query_snippet).execute()
        df = pd.DataFrame(response.data)
        return df.to_markdown()
    except Exception as e:
        return f"Error running query: {e}"



final_answer = FinalAnswerTool()


# Initialize model with validated parameters
def create_model():
    return HfApiModel(
        max_tokens=2096,
        temperature=0.5,
        model_id='meta-llama/Llama-3.1-8B-Instruct',
        custom_role_conversions=None
    )

def main():
    model = create_model()
    
    with open("prompts.yaml", 'r') as stream:
        prompt_templates = yaml.safe_load(stream)
    
    # Create agent with validated components
    agent = CodeAgent(
        model=model,
        tools=[final_answer, select],
        max_steps=6,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name="SQLAgent",  # Add a name for better identification
        description="An agent that can execute SQL queries and provide answers",
        prompt_templates=prompt_templates,
    )

    # Launch the UI
    GradioUI(agent).launch()

if __name__ == "__main__":
    main()