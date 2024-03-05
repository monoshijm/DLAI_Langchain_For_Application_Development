# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:51:15 2024

@author: mmajumdar1
"""


"""
LangChain - Memory
Conversation Memory
1. Token Memory
2. Buffer Window
3. Token Buffer

"""
!pip install openai
!pip install langchain
# Set parameters

import os
import openai
import langchain
from langchain.llms import AzureOpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI    
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory



os.environ["OPENAI_API_TYPE"]="azure"
os.environ["OPENAI_API_BASE"]="https://genaipoc-kgs-ds-and-ai-team.openai.azure.com/"
os.environ["OPENAI_API_VERSION"]="2023-03-15-preview"
os.environ["OPENAI_API_KEY"]="2926864d111c4d7d80d96c1d605921e3"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base =  os.environ["OPENAI_API_BASE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]
openai.api_type = os.environ["OPENAI_API_TYPE"]

llm_model = "gpt-3.5-turbo-16k"
engine = "gpt-35-turbo-16k"


llm = ChatOpenAI(temperature=0.0, model=llm_model,engine=engine)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Andrew")

conversation.predict(input="What is 1+1?")

conversation.predict(input="What is my name?")

print(memory.buffer)

memory.load_memory_variables({})

memory = ConversationBufferMemory()

memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})

print(memory.buffer)

memory.load_memory_variables({})

memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})

memory.load_memory_variables({})

# Conversation Buffer Window Memory

from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1)               
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.load_memory_variables({})

llm = ChatOpenAI(temperature=0.0, model=llm_model,engine=engine)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)

conversation.predict(input="Hi, my name is Andrew")

conversation.predict(input="What is 1+1?")

conversation.predict(input="What is my name?")


# Conversation Token Buffer Memory

from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0, model=llm_model,engine = engine)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

memory.load_memory_variables({})

# Conversation Summary Memory

from langchain.memory import ConversationSummaryBufferMemory

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

memory.load_memory_variables({})

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="What would be a good demo to show?")

memory.load_memory_variables({})





