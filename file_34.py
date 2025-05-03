#!/usr/bin/env python
# coding: utf-8

# # Double Texting
# 
# Seamless handling of [double texting](https://langchain-ai.github.io/langgraph/concepts/double_texting/) is important for handling real-world usage scenarios, especially in chat applications.
# 
# Users can send multiple messages in a row before the prior run(s) complete, and we want to ensure that we handle this gracefully.
# 
# ## Reject
# 
# A simple approach is to [reject](https://langchain-ai.github.io/langgraph/cloud/how-tos/reject_concurrent/) any new runs until the current run completes.

# In[13]:


get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install -U langgraph_sdk\n')


# In[2]:


from langgraph_sdk import get_client
url_for_cli_deployment = "http://localhost:8123"
client = get_client(url=url_for_cli_deployment)


# In[18]:


import httpx
from langchain_core.messages import HumanMessage

# Create a thread
thread = await client.threads.create()

# Create to dos
user_input_1 = "Add a ToDo to follow-up with DI Repairs."
user_input_2 = "Add a ToDo to mount dresser to the wall."
config = {"configurable": {"user_id": "Test-Double-Texting"}}
graph_name = "task_maistro" 

run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]}, 
    config=config,
)
try:
    await client.runs.create(
        thread["thread_id"],
        graph_name,
        input={"messages": [HumanMessage(content=user_input_2)]}, 
        config=config,
        multitask_strategy="reject",
    )
except httpx.HTTPStatusError as e:
    print("Failed to start concurrent run", e)


# In[19]:


from langchain_core.messages import convert_to_messages

# Wait until the original run completes
await client.runs.join(thread["thread_id"], run["run_id"])

# Get the state of the thread
state = await client.threads.get_state(thread["thread_id"])
for m in convert_to_messages(state["values"]["messages"]):
    m.pretty_print()


# ## Enqueue
# 
# We can use [enqueue](https://langchain-ai.github.io/langgraph/cloud/how-tos/enqueue_concurrent/https://langchain-ai.github.io/langgraph/cloud/how-tos/enqueue_concurrent/) any new runs until the current run completes.

# In[20]:


# Create a new thread
thread = await client.threads.create()

# Create new ToDos
user_input_1 = "Send Erik his t-shirt gift this weekend."
user_input_2 = "Get cash and pay nanny for 2 weeks. Do this by Friday."
config = {"configurable": {"user_id": "Test-Double-Texting"}}
graph_name = "task_maistro" 

first_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]}, 
    config=config,
)

second_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_2)]}, 
    config=config,
    multitask_strategy="enqueue",
)

# Wait until the second run completes
await client.runs.join(thread["thread_id"], second_run["run_id"])

# Get the state of the thread
state = await client.threads.get_state(thread["thread_id"])
for m in convert_to_messages(state["values"]["messages"]):
    m.pretty_print()


# ## Interrupt
# 
# We can use [interrupt](https://langchain-ai.github.io/langgraph/cloud/how-tos/interrupt_concurrent/) to interrupt the current run, but save all the work that has been done so far up to that point.
# 

# In[32]:


import asyncio

# Create a new thread
thread = await client.threads.create()

# Create new ToDos
user_input_1 = "Give me a summary of my ToDos due tomrrow."
user_input_2 = "Never mind, create a ToDo to Order Ham for Thanksgiving by next Friday."
config = {"configurable": {"user_id": "Test-Double-Texting"}}
graph_name = "task_maistro" 

interrupted_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]}, 
    config=config,
)

# Wait for some of run 1 to complete so that we can see it in the thread 
await asyncio.sleep(1)

second_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_2)]}, 
    config=config,
    multitask_strategy="interrupt",
)

# Wait until the second run completes
await client.runs.join(thread["thread_id"], second_run["run_id"])

# Get the state of the thread
state = await client.threads.get_state(thread["thread_id"])
for m in convert_to_messages(state["values"]["messages"]):
    m.pretty_print()


# We can see the initial run is saved, and has status `interrupted`.

# In[33]:


# Confirm that the first run was interrupted
print((await client.runs.get(thread["thread_id"], interrupted_run["run_id"]))["status"])


# ## Rollback
# 
# We can use [rollback](https://langchain-ai.github.io/langgraph/cloud/how-tos/rollback_concurrent/) to interrupt the prior run of the graph, delete it, and start a new run with the double-texted input.
# 

# In[28]:


# Create a new thread
thread = await client.threads.create()

# Create new ToDos
user_input_1 = "Add a ToDo to call to make appointment at Yoga."
user_input_2 = "Actually, add a ToDo to drop by Yoga in person on Sunday."
config = {"configurable": {"user_id": "Test-Double-Texting"}}
graph_name = "task_maistro" 

rolled_back_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_1)]}, 
    config=config,
)

second_run = await client.runs.create(
    thread["thread_id"],
    graph_name,
    input={"messages": [HumanMessage(content=user_input_2)]}, 
    config=config,
    multitask_strategy="rollback",
)

# Wait until the second run completes
await client.runs.join(thread["thread_id"], second_run["run_id"])

# Get the state of the thread
state = await client.threads.get_state(thread["thread_id"])
for m in convert_to_messages(state["values"]["messages"]):
    m.pretty_print()


# The initial run was deleted.

# In[29]:


# Confirm that the original run was deleted
try:
    await client.runs.get(thread["thread_id"], rolled_back_run["run_id"])
except httpx.HTTPStatusError as _:
    print("Original run was correctly deleted")


# ### Summary 
# 
# We can see [all the methods summarized](https://langchain-ai.github.io/langgraph/concepts/double_texting/):
# 
# ![Screenshot 2024-11-15 at 12.13.18 PM.png](attachment:ff0af98b-71b1-497a-9c0e-b3519662fd2c.png)
