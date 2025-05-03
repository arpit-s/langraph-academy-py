#!/usr/bin/env python
# coding: utf-8

# # Connecting to a LangGraph Platform Deployment
# 
# ## Deployment Creation
# 
# We just created a [deployment](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph) for the `task_maistro` app from module 5.
# 
# * We used the [the LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/#commands) to build a Docker image for the LangGraph Server with our `task_maistro` graph.
# * We used the provided `docker-compose.yml` file to create three separate containers based on the services defined: 
#     * `langgraph-redis`: Creates a new container using the official Redis image.
#     * `langgraph-postgres`: Creates a new container using the official Postgres image.
#     * `langgraph-api`: Creates a new container using our pre-built `task_maistro` Docker image.
# 
# ```
# $ cd module-6/deployment
# $ docker compose up
# ```
# 
# Once running, we can access the deployment through:
#       
# * API: http://localhost:8123
# * Docs: http://localhost:8123/docs
# * LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123
# 
# ![langgraph-platform-high-level.png](attachment:3a5ede4f-7a62-4e05-9e44-301465ca8555.png)
# 
# ## Using the API  
# 
# LangGraph Server exposes [many API endpoints](https://github.com/langchain-ai/agent-protocol) for interacting with the deployed agent.
# 
# We can group [these endpoints into a few common agent needs](https://github.com/langchain-ai/agent-protocol): 
# 
# * **Runs**: Atomic agent executions
# * **Threads**: Multi-turn interactions or human in the loop
# * **Store**: Long-term memory
# 
# We can test requests directly [in the API docs](http://localhost:8123/docs#tag/thread-runs).

# ## SDK
# 
# The [LangGraph SDKs](https://langchain-ai.github.io/langgraph/concepts/sdk/) (Python and JS) provide a developer-friendly interface to interact with the LangGraph Server API presented above.

# In[ ]:


get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install -U langgraph_sdk\n')


# In[3]:


from langgraph_sdk import get_client

# Connect via SDK
url_for_cli_deployment = "http://localhost:8123"
client = get_client(url=url_for_cli_deployment)


# ## Remote Graph
# 
# If you are working in the LangGraph library, [Remote Graph](https://langchain-ai.github.io/langgraph/how-tos/use-remote-graph/) is also a useful way to connect directly to the graph.

# In[8]:


get_ipython().run_cell_magic('capture', '--no-stderr', '%pip install -U langchain_openai langgraph langchain_core\n')


# In[4]:


from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import convert_to_messages
from langchain_core.messages import HumanMessage, SystemMessage

# Connect via remote graph
url = "http://localhost:8123"
graph_name = "task_maistro" 
remote_graph = RemoteGraph(graph_name, url=url)


# ## Runs
# 
# A "run" represents a [single execution](https://github.com/langchain-ai/agent-protocol?tab=readme-ov-file#runs-atomic-agent-executions) of your graph. Each time a client makes a request:
# 
# 1. The HTTP worker generates a unique run ID
# 2. This run and its results are stored in PostgreSQL
# 3. You can query these runs to:
#    - Check their status
#    - Get their results
#    - Track execution history
# 
# You can see a full set of How To guides for various types of runs [here](https://langchain-ai.github.io/langgraph/how-tos/#runs).
# 
# Let's looks at a few of the interesting things we can do with runs.
# 
# ### Background Runs
# 
# The LangGraph server supports two types of runs: 
# 
# * `Fire and forget` - Launch a run in the background, but don’t wait for it to finish
# * `Waiting on a reply (blocking or polling)` - Launch a run and wait/stream its output
# 
# Background runs and polling are quite useful when working with long-running agents. 
# 
# Let's [see](https://langchain-ai.github.io/langgraph/cloud/how-tos/background_run/#check-runs-on-thread) how this works:

# In[6]:


# Create a thread
thread = await client.threads.create()
thread


# In[7]:


# Check any existing runs on a thread
thread = await client.threads.create()
runs = await client.runs.list(thread["thread_id"])
print(runs)


# In[8]:


# Ensure we've created some ToDos and saved them to my user_id
user_input = "Add a ToDo to finish booking travel to Hong Kong by end of next week. Also, add a ToDo to call parents back about Thanksgiving plans."
config = {"configurable": {"user_id": "Test"}}
graph_name = "task_maistro" 
run = await client.runs.create(thread["thread_id"], graph_name, input={"messages": [HumanMessage(content=user_input)]}, config=config)


# In[9]:


# Kick off a new thread and a new run
thread = await client.threads.create()
user_input = "Give me a summary of all ToDos."
config = {"configurable": {"user_id": "Test"}}
graph_name = "task_maistro" 
run = await client.runs.create(thread["thread_id"], graph_name, input={"messages": [HumanMessage(content=user_input)]}, config=config)


# In[10]:


# Check the run status
print(await client.runs.get(thread["thread_id"], run["run_id"]))


# We can see that it has `'status': 'pending'` because it is still running.
# 
# What if we want to wait until the run completes, making it a blocking run?
# 
# We can use `client.runs.join` to wait until the run completes.
# 
# This ensures that no new runs are started until the current run completes on the thread.

# In[11]:


# Wait until the run completes
await client.runs.join(thread["thread_id"], run["run_id"])
print(await client.runs.get(thread["thread_id"], run["run_id"]))


# Now the run has `'status': 'success'` because it has completed.

# ### Streaming Runs
# 
# Each time a client makes a streaming request:
# 
# 1. The HTTP worker generates a unique run ID
# 2. The Queue worker begins work on the run 
# 3. During execution, the Queue worker publishes update to Redis
# 4. The HTTP worker subscribes to updates from Redis for ths run, and returns them to the client 
# 
# This enabled streaming! 
# 
# We've covered [streaming](https://langchain-ai.github.io/langgraph/how-tos/#streaming_1) in previous modules, but let's pick one method -- streaming tokens -- to highlight.
# 
# Streaming tokens back to the client is especially useful when working with production agents that may take a while to complete.
# 
# We [stream tokens](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/#setup) using `stream_mode="messages-tuple"`.

# In[12]:


user_input = "What ToDo should I focus on first."
async for chunk in client.runs.stream(thread["thread_id"], 
                                      graph_name, 
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      config=config,
                                      stream_mode="messages-tuple"):

    if chunk.event == "messages":
        print("".join(data_item['content'] for data_item in chunk.data if 'content' in data_item), end="", flush=True)


# ## Threads
# 
# Whereas a run is only a single execution of the graph, a thread supports *multi-turn* interactions.
# 
# When the client makes a graph execution execution with a `thread_id`, the server will save all [checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints) (steps) in the run to the thread in the Postgres database.
# 
# The server allows us to [check the status of created threads](https://langchain-ai.github.io/langgraph/cloud/how-tos/check_thread_status/).
# 
# ### Check thread state
# 
# In addition, we can easily access the state [checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints) saved to any specific thread.

# In[13]:


thread_state = await client.threads.get_state(thread['thread_id'])
for m in convert_to_messages(thread_state['values']['messages']):
    m.pretty_print()


# ### Copy threads
# 
# We can also [copy](https://langchain-ai.github.io/langgraph/cloud/how-tos/copy_threads/) (i.e. "fork") an existing thread. 
# 
# This will keep the existing thread's history, but allow us to create independent runs that do not affect the original thread.

# In[14]:


# Copy the thread
copied_thread = await client.threads.copy(thread['thread_id'])


# In[15]:


# Check the state of the copied thread
copied_thread_state = await client.threads.get_state(copied_thread['thread_id'])
for m in convert_to_messages(copied_thread_state['values']['messages']):
    m.pretty_print()


# ### Human in the loop
# 
# We covered [Human in the loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/) in Module 3, and the server supports all Human in the loop features that we discussed.
# 
# As an example, [we can search, edit, and continue graph execution](https://langchain-ai.github.io/langgraph/concepts/persistence/#capabilities) from any prior checkpoint. 

# In[16]:


# Get the history of the thread
states = await client.threads.get_history(thread['thread_id'])

# Pick a state update to fork
to_fork = states[-2]
to_fork['values']


# In[17]:


to_fork['values']['messages'][0]['id']


# In[18]:


to_fork['next']


# In[19]:


to_fork['checkpoint_id']


# Let's edit the state. Remember how our reducer on `messages` works: 
# 
# * It will append, unless we supply a message ID.
# * We supply the message ID to overwrite the message, rather than appending to state!

# In[20]:


forked_input = {"messages": HumanMessage(content="Give me a summary of all ToDos that need to be done in the next week.",
                                         id=to_fork['values']['messages'][0]['id'])}

# Update the state, creating a new checkpoint in the thread
forked_config = await client.threads.update_state(
    thread["thread_id"],
    forked_input,
    checkpoint_id=to_fork['checkpoint_id']
)


# In[21]:


# Run the graph from the new checkpoint in the thread
async for chunk in client.runs.stream(thread["thread_id"], 
                                      graph_name, 
                                      input=None,
                                      config=config,
                                      checkpoint_id=forked_config['checkpoint_id'],
                                      stream_mode="messages-tuple"):

    if chunk.event == "messages":
        print("".join(data_item['content'] for data_item in chunk.data if 'content' in data_item), end="", flush=True)


# ## Across-thread memory
# 
# In module 5, we covered how the [LangGraph memory `store`](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store) can be used to save information across threads.
# 
# Our deployed graph, `task_maistro`, uses the `store` to save information -- such as ToDos -- namespaced to the `user_id`.
# 
# Our deployment includes a Postgres database, which stores these long-term (across-thread) memories.
# 
# There are several methods available [for interacting with the store](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient) in our deployment using the LangGraph SDK.
# 
# ### Search items
# 
# The `task_maistro` graph uses the `store` to save ToDos namespaced by default to (`todo`, `todo_category`, `user_id`). 
# 
# The `todo_category` is by default set to `general` (as you can see in `deployment/configuration.py`).
# 
# We can simply supply this tuple to search for all ToDos. 

# In[29]:


items = await client.store.search_items(
    ("todo", "general", "Test"),
    limit=5,
    offset=0
)
items['items']


# ### Add items
# 
# In our graph, we call `put` to add items to the store.
# 
# We can use [put](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient.put_item) with the SDK if we want to directly add items to the store outside our graph.

# In[30]:


from uuid import uuid4
await client.store.put_item(
    ("testing", "Test"),
    key=str(uuid4()),
    value={"todo": "Test SDK put_item"},
)


# In[31]:


items = await client.store.search_items(
    ("testing", "Test"),
    limit=5,
    offset=0
)
items['items']


# ### Delete items
# 
# We can use the SDK to [delete items](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient.delete_item) from the store by key.

# In[32]:


[item['key'] for item in items['items']]


# In[33]:


await client.store.delete_item(
       ("testing", "Test"),
        key='3de441ba-8c79-4beb-8f52-00e4dcba68d4',
    )


# In[34]:


items = await client.store.search_items(
    ("testing", "Test"),
    limit=5,
    offset=0
)
items['items']


# 
