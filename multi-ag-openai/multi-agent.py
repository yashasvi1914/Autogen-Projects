import asyncio
import os
import streamlit as st
from dotenv import load_dotenv

# AutoGen 0.4+ Imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination    
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


'''Text Mention Termination just means they'll stop the round robin game once someone says the word terminate.
 The RoundRobin group chat part is just the conductor, deciding whose turn it is, 
 first analyst, then Python developer, then reviewer.'''


# Page Config
st.set_page_config(page_title="Multi-Agent Team by YASHASVI CHOUDHARY - AutoGen 0.4 (OpenAI)", page_icon="👥", layout="wide")

load_dotenv()

st.title("👥 Multi-Agent Collaboration Team (OpenAI)")
st.caption("Watch 3 agents (Analyst -> Developer -> Reviewer) collaborate using GPT-4o")

# 1. API Key Check
openai_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_key:
    st.warning("Please provide an `OPENAI_API_KEY` in your `.env` file or sidebar.")
    st.stop()

# 2. Input Task
user_task = st.text_area(
    "Enter a software or business task:",
    value="Create a Python script to calculate monthly EMI for a loan and print a summary table.",
    height=100
)

# 3. Async Multi-Agent Core Function
async def run_multi_agent_team(task: str, api_key: str):
    # Initialize OpenAI Client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=api_key,
    )

    # Agent 1: Business Analyst
    analyst = AssistantAgent(
        name="Business_Analyst",
        model_client=model_client,
        system_message="You are a Business Analyst. Break user requests into 3 clear technical requirements.",
    )

    # Agent 2: Developer
    developer = AssistantAgent(
        name="Python_Developer",
        model_client=model_client,
        system_message="You are a Senior Python Developer. Write clean Python code following the analyst's requirements.",
    )

    # Agent 3: Reviewer
    reviewer = AssistantAgent(
        name="Code_Reviewer",
        model_client=model_client,
        system_message=(
            "You are a Code Reviewer. Review the Python code. "
            "If acceptable, reply with 'APPROVED' and 'TERMINATE'. Otherwise, request changes."
        ),
    )

    # Set up group chat with termination condition
    team = RoundRobinGroupChat(
        participants=[analyst, developer, reviewer],
        termination_condition=TextMentionTermination("TERMINATE"),
        max_turns=6,
    )

    # Capture turn-by-turn messages from stream
    messages = []
    async for message in team.run_stream(task=task):
        messages.append(message)

    await model_client.close()
    return messages

# 4. Trigger Team Execution
if st.button("🚀 Run Multi-Agent Team", type="primary"):
    with st.spinner("Team is collaborating..."):
        try:
            stream_messages = asyncio.run(run_multi_agent_team(user_task, openai_key))

            st.subheader("💬 Live Team Conversation Log")
            for msg in stream_messages:
                if hasattr(msg, "source") and hasattr(msg, "content"):
                    with st.chat_message(msg.source):
                        st.write(f"**Agent: {msg.source}**")
                        st.markdown(msg.content)
        except Exception as e:
            st.error(f"Error executing multi-agent team: {e}")