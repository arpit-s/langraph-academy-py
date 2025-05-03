#!/usr/bin/env python
# coding: utf-8

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-3/edit-state-human-feedback.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239520-lesson-3-editing-state-and-human-feedback)

# # Editing graph state
# 
# ## Review
# 
# We discussed motivations for human-in-the-loop:
# 
# (1) `Approval` - We can interrupt our agent, surface state to a user, and allow the user to accept an action
# 
# (2) `Debugging` - We can rewind the graph to reproduce or avoid issues
# 
# (3) `Editing` - You can modify the state 
# 
# We showed how breakpoints support user approval, but don't yet know how to modify our graph state once our graph is interrupted!
# 
# ## Goals
# 
# Now, let's show how to directly edit the graph state and insert human feedback.

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install --quiet -U langgraph langchain_openai langgraph_sdk langgraph-prebuilt\n')


# In[ ]:


import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")


# ## Editing state 
# 
# Previously, we introduced breakpoints.
# 
# We used them to interrupt the graph and await user approval before executing the next node.
# 
# But breakpoints are also [opportunities to modify the graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/).
# 
# Let's set up our agent with a breakpoint before the `assistant` node.

# In[1]:


from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)


# In[2]:


from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)

# Show
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))


# Let's run!
# 
# We can see the graph is interrupted before the chat model responds. 

# In[3]:


# Input
initial_input = {"messages": "Multiply 2 and 3"}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()


# In[4]:


state = graph.get_state(thread)
state


# Now, we can directly apply a state update.
# 
# Remember, updates to the `messages` key will use the `add_messages` reducer:
#  
# * If we want to over-write the existing message, we can supply the message `id`.
# * If we simply want to append to our list of messages, then we can pass a message without an `id` specified, as shown below.

# In[5]:


graph.update_state(
    thread,
    {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
)


# Let's have a look.
# 
# We called `update_state` with a new message. 
# 
# The `add_messages` reducer appends it to our state key, `messages`.

# In[6]:


new_state = graph.get_state(thread).values
for m in new_state['messages']:
    m.pretty_print()


# Now, let's proceed with our agent, simply by passing `None` and allowing it proceed from the current state.
# 
# We emit the current and then proceed to execute the remaining nodes.

# In[7]:


for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()


# Now, we're back at the `assistant`, which has our `breakpoint`.
# 
# We can again pass `None` to proceed.

# In[8]:


for event in graph.stream(None, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()


# ### Editing graph state in Studio
# 
# **⚠️ DISCLAIMER**
# 
# Since the filming of these videos, we've updated Studio so that it can be run locally and opened in your browser. This is now the preferred way to run Studio (rather than using the Desktop App as shown in the video). See documentation [here](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server) on the local development server and [here](https://langchain-ai.github.io/langgraph/how-tos/local-studio/#run-the-development-server). To start the local development server, run the following command in your terminal in the `/studio` directory in this module:
# 
# ```
# langgraph dev
# ```
# 
# You should see the following output:
# ```
# - 🚀 API: http://127.0.0.1:2024
# - 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
# - 📚 API Docs: http://127.0.0.1:2024/docs
# ```
# 
# Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.
# 
# The LangGraph API [supports editing graph state](https://langchain-ai.github.io/langgraph/cloud/how-tos/human_in_the_loop_edit_state/#initial-invocation). 

# In[ ]:


if 'google.colab' in str(get_ipython()):
    raise Exception("Unfortunately LangGraph Studio is currently not supported on Google Colab")


# In[9]:


# This is the URL of the local development server
from langgraph_sdk import get_client
client = get_client(url="http://127.0.0.1:2024")


# Our agent is defined in `studio/agent.py`. 
# 
# If you look at the code, you'll see that it *does not* have a breakpoint! 
#  
# Of course, we can add it to `agent.py`, but one very nice feature of the API is that we can pass in a breakpoint!
# 
# Here, we pass a `interrupt_before=["assistant"]`.

# In[10]:


initial_input = {"messages": "Multiply 2 and 3"}
thread = await client.threads.create()
async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input=initial_input,
    stream_mode="values",
    interrupt_before=["assistant"],
):
    print(f"Receiving new event of type: {chunk.event}...")
    messages = chunk.data.get('messages', [])
    if messages:
        print(messages[-1])
    print("-" * 50)


# We can get the current state

# In[11]:


current_state = await client.threads.get_state(thread['thread_id'])
current_state


# We can look at the last message in state.

# In[12]:


last_message = current_state['values']['messages'][-1]
last_message


# We can edit it!

# In[13]:


last_message['content'] = "No, actually multiply 3 and 3!"
last_message


# In[14]:


last_message


# Remember, as we said before, updates to the `messages` key will use the same `add_messages` reducer. 
# 
# If we want to over-write the existing message, then we can supply the message `id`.
# 
# Here, we did that. We only modified the message `content`, as shown above.

# In[15]:


await client.threads.update_state(thread['thread_id'], {"messages": last_message})


# Now, we resume by passing `None`. 

# In[16]:


async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=None,
    stream_mode="values",
    interrupt_before=["assistant"],
):
    print(f"Receiving new event of type: {chunk.event}...")
    messages = chunk.data.get('messages', [])
    if messages:
        print(messages[-1])
    print("-" * 50)


# We get the result of the tool call as `9`, as expected.

# In[17]:


async for chunk in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input=None,
    stream_mode="values",
    interrupt_before=["assistant"],
):
    print(f"Receiving new event of type: {chunk.event}...")
    messages = chunk.data.get('messages', [])
    if messages:
        print(messages[-1])
    print("-" * 50)


# ## Awaiting user input
# 
# So, it's clear that we can edit our agent state after a breakpoint.
# 
# Now, what if we want to allow for human feedback to perform this state update?
# 
# We'll add a node that [serves as a placeholder for human feedback](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/#setup) within our agent.
# 
# This `human_feedback` node allow the user to add feedback directly to state.
#  
# We specify the breakpoint using `interrupt_before` our `human_feedback` node.
# 
# We set up a checkpointer to save the state of the graph up until this node.

# In[18]:


# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# no-op node that should be interrupted on
def human_feedback(state: MessagesState):
    pass

# Assistant node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

# Define edges: these determine the control flow
builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "human_feedback")

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))


# We will get feedback from the user.
# 
# We use `.update_state` to update the state of the graph with the human response we get, as before.
# 
# We use the `as_node="human_feedback"` parameter to apply this state update as the specified node, `human_feedback`.

# In[19]:


# Input
initial_input = {"messages": "Multiply 2 and 3"}

# Thread
thread = {"configurable": {"thread_id": "5"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()
    
# Get user input
user_input = input("Tell me how you want to update the state: ")

# We now update the state as if we are the human_feedback node
graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")

# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()


# In[20]:


# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()

