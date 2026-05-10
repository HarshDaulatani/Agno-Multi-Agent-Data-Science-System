from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.os import AgentOS
from agno.team import Team
from agno.db.sqlite import SqliteDb
from agno.tools.csv_toolkit import CsvTools
from agno.tools.file import FileTools
from agno.tools.pandas import PandasTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools
from agno.tools.visualization import VisualizationTools
from dotenv import load_dotenv
from pathlib import Path


# load the api keys
load_dotenv()

# path for the base/project directory
base_dir = Path(__file__).parent

# path for the data 
data_path = Path(__file__).parent / "data" / "car_details.csv"

# build the database
db = SqliteDb(db_file="memory.db",
              session_table="session_table")

# create the model
model = OpenAIChat(id="gpt-5-mini")


# ============================ Agents ===============================

# create the data loader agent
data_loader_agent = Agent(
    id="data-loader-agent",
    name="Data Loader Agent",
    model=model,
    role="Data loading and reading csv files",
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    instructions=["You are an expert Data Loader Agent responsible for reading CSV files from the project directory.",
                  "Your primary capability is to list and preview CSV files effectively.",
                  "To prevent overwhelming the context window, never read more than 20 to 30 rows of a CSV file at once.",
                  "You can search for specific files within the project folder when requested.",
                  "You have access to file management tools to list, read, and write files as necessary.",
                  "Ensure you strictly use the provided CSV tool when interacting with CSV files."],
    tools=[CsvTools(enable_query_csv_file=False, csvs=[data_path]),
           FileTools(base_dir=base_dir)],
)


# file manager agent
file_manager_agent = Agent(
    id="file-manager-agent",
    name="File Manager Agent",
    model=model,
    role="Manages Filesystem",
    instructions=["You are an expert File Management Agent dedicated to navigating and managing the project's filesystem.",
                  "Your main responsibility is to list the contents of directories and locate specific files when asked.",
                  "You are fully capable of reading from and writing to standard files.",
                  "Under no circumstances should you read the contents of CSV files; you are only permitted to list them."],
    tools=[FileTools(base_dir=base_dir)]
)


# data understanding agent
data_understanding_agent = Agent(
    id="data-understanding-agent",
    name="Data Understanding Agent",
    model=model,
    role="Data understanding and Exploration assistant",
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    instructions=["You are an expert Data Understanding Agent proficient in pandas and exploratory data analysis (EDA).",
                  "You can seamlessly create pandas DataFrames and apply various analytical operations to them.",
                  "Routinely use operations such as .head(), .tail(), .info(), and .describe() for numerical summaries, and .value_counts() for categorical distributions.",
                  "Always explicitly identify and list the numerical and categorical columns present in the dataset.",
                  "Regularly check and report the overall shape of the DataFrame using the .shape attribute.",
                  "You have integrated file tools to independently search for and locate data files within the workspace."
                  ],
    tools=[PandasTools(), FileTools(base_dir=base_dir)],
)


# create a visualization agent
visualization_agent = Agent(
    id="viz-agent",
    name="Visualization Agent",
    model=model,
    role="Plotting and Visualization Assistant",
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    tools=[PandasTools(), FileTools(base_dir=base_dir), VisualizationTools("plots")],
    instructions=["You are an expert Visualization Agent specializing in creating insightful plots using matplotlib.",
                  "You can use file tools to discover and list data files within the project directory.",
                  "Leverage your pandas tools to process and structure the data properly before passing it into visualization functions.",
                  "You are skilled at generating bar plots, pie charts, line plots, histograms, and scatter plots.",
                  "Prefer bar plots for univariate categorical data and histograms for univariate numerical data.",
                  "When analyzing the relationship between two numerical variables, utilize scatter plots.",
                  "Always verify that you are feeding the correct data format and variables into the respective chart types."]
)


# create a coding agent
coding_agent = Agent(
    id="coding-agent",
    name="Coding Agent",
    db=db,
    add_history_to_context=True,
    role="Python Coding Assistant",
    num_history_runs=5,
    read_chat_history=True,
    model=model,
    tools=[DuckDuckGoTools(), PythonTools(base_dir=base_dir), ShellTools(base_dir=base_dir)],
    instructions=["You are an expert Python Coding Agent specializing in Machine Learning and Data Science solutions.",
                  "Your primary directive is to write clean, efficient, and robust code for ML workflows.",
                  "You are highly proficient in utilizing core libraries such as numpy, pandas, scikit-learn, and scipy.",
                  "You can freely navigate the directory to list and read relevant files.",
                  "Use your Python tools to create, write to, and execute Python scripts to verify your output.",
                  "Your expertise covers data cleaning, advanced feature engineering, model training, and rigorous evaluation.",
                  "Utilize your web search capabilities via DuckDuckGo to look up the latest library documentation or resolve technical issues.",
                  "Always present your proposed code to the user for review; only write it to a file after receiving explicit approval.",
                  "If you need to install new packages, strictly use the shell tool with the command `uv add <package_name>`.",
                  "Do not use the shell tool for any other commands outside of package installation."]
)


# create the shell agent
shell_agent = Agent(
    id="shell-agent",
    name="Shell Agent",
    model=model,
    db=db,
    add_history_to_context=True,
    tools=[ShellTools(base_dir=base_dir)],
    role="Shell Commands Executor",
    instructions=["You are a specialized Shell Agent tasked with safely executing terminal commands.",
                  "Exercise extreme caution when using your shell capabilities.",
                  "Only resort to the shell tool if the standard Python tools fail to execute a script properly.",
                  "When running a python file, strictly use the command format: `uv run <python_file.py>`.",
                  "Never execute commands that modify, restructure, or delete the project files or directories.",
                  "Under no circumstances should you delete any files.",
                  "Restrict your usage to reading the project structure, verifying files, or securely executing scripts."]
)

# ======================= Team ======================================

# create the team
data_science_team = Team(
    members=[data_loader_agent,
             file_manager_agent,
             data_understanding_agent,
             visualization_agent,
             coding_agent,
             shell_agent],
    id="data-science-team",
    name="Data Science Team",
    role="Team Leader / Project Manager",
    model=model,
    instructions=["You are an expert Data Scientist and the Team Leader orchestrating a multi-agent AI system.",
                  "You manage specialized agents highly skilled in coding, pandas manipulation, matplotlib visualization, shell execution, data ingestion, and filesystem management.",
                  "Efficiently delegate tasks to the appropriate team members based on their specific roles and capabilities.",
                  "Your goal is to guide the user through a comprehensive ML pipeline: data loading, EDA, cleaning, feature engineering, modeling, and evaluation.",
                  "Before saving or executing any generated code, always present it to the user for review and approval.",
                  "Provide clear, step-by-step guidance so the user always understands the current stage and what needs to happen next.",
                  "Leverage your session state memory to store and retrieve important context throughout the conversation.",
                  "Proactively save critical project variables like data paths, model paths, and source code directories to the session state.",
                  "Whenever the user asks you to remember a specific detail, immediately store it in the session state.",
                  "Actively propose innovative ideas, analytical approaches, and best practices like a Senior Data Scientist.",
                  "If errors occur, adopt a structured, step-by-step debugging approach to identify and resolve the root cause."],
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    read_chat_history=True,
    session_state={},
    add_session_state_to_context=True,
    add_member_tools_to_context=True,
    enable_agentic_state=True
)

# ==================== Agent OS =======================================
agent_os = AgentOS(
    id='agent-os',
    name="Data Science Team",
    description="This team of agents helps you throughout your data science and ML pipeline journey",
    teams=[data_science_team]
)


# create the fastapi app
app = agent_os.get_app()

if __name__ == "__main__":
    # serve the app
    agent_os.serve(
        app="app:app"
    )