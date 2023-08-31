# Headline: Flexible and Easy Conversational AI with Streamlit powered by Clarifai
## By just changing the plain English prompt, quickly experiment with different LLMs and applications

Author: Ian Kelk, Product Marketing Manager, Clarifai

👉 **TLDR:** This blog post showcases how to build an engaging and versatile chatbot using the Clarifai API and Streamlit. **Links:** Here's the [app](https://llm-text-adventure.streamlit.app/) and the [code](https://github.com/iankelk/llm-text-adventure).

This Streamlit app lets you chat with several Large Language Models. It has two main capabilities:
- It proves how powerful and easy it is to integrate models provided by Clarifai using Streamlit & Langchain
- You can evaluate the responses from multiple LLMs and choose the one which best suits your purpose.
- You can see how just by changing the initial prompt to the LLM, you can completely change the entire nature of the app.

https://llm-text-adventure.streamlit.app

## Introduction
Hello, Streamlit Community! 👋 I'm Ian Kelk, a machine learning enthusiast and Developer Relations Manager at Clarifai. My journey into data science began with a strong fascination for AI and its applications, particularly within the lens of natural language processing.

## Problem statement
It can seem intimidating to have to create an entirely new Streamlit app every time you find a new use case for an LLM. It also requires knowing a decent amount of Python and the Streamlit API. What if, instead, we can create completely different apps *just* by changing the prompt? This requires nearly zero programming or expertise, and the results can be surprisingly good. In response to this, I've created a Streamlit chatbot application of sorts, that works with a hidden starting prompt that can radically change its behavour. It combines the interactivity of Streamlit's features with the intelligence of Clarifai's models.

In this post, you’ll learn how to build an AI-powered Chatbot:

Step 1: Create the environment to work with Streamlit locally

Step 2: Create the Secrets File and define the Prompt

Step 3: Set Up the Streamlit App

Step 4: Deploy the app on Streamlit's cloud.


## App overview / Technical details
The application integrates the Clarifai API with a Streamlit interface. Clarifai is known for its superb artificial intelligence models, while Streamlit provides an elegant framework for user interaction. Using a secrets.toml file for secure handling of the Clarifai Personal Authentication Token (PAT) and additional settings, the application allows users to interact with different Language Learning Models (LLMs) using a chat interface. The secret sauce however, is the inclusion of a separate `prompts.py` file which allows for different behaviour of the application purely based on the prompt.

Let's take a look at the app in action:

![](./assets/italian.gif)

**Step A**

As with any Python project, it's always best to create a virtual environment. Here's how to create a virtual environment named `llm-text-adventure` using both `conda` and `venv` in Linux:

### 1. Using `conda`:

1. **Create the virtual environment:**
   
   ```bash
   conda create --name llm-text-adventure python=3.8
   ```

   Note: Here, I'm specifying Python 3.8 as an example. You can replace it with your desired version.

2. **Activate the virtual environment:**

   ```bash
   conda activate llm-text-adventure
   ```

3. When you're done and wish to deactivate the environment:

   ```bash
   conda deactivate
   ```

### 2. Using `venv`:

1. First, ensure you have `venv` module installed. If not, install the required version of Python which includes `venv` by default. If you have Python 3.3 or newer, `venv` should be included.

2. **Create the virtual environment:**

   ```bash
   python3 -m venv llm-text-adventure
   ```

   Note: You may need to replace `python3` with just `python` or another specific version, depending on your system setup.

3. **Activate the virtual environment:**

   ```bash
   source llm-text-adventure/bin/activate
   ```

   When the environment is activated, you'll see the environment name (`llm-text-adventure`) at the beginning of your command prompt.

4. To deactivate the virtual environment and return to the global Python environment:

   ```bash
   deactivate
   ```

That's it! Depending on your project requirements and the tools you're familiar with, you can choose either `conda` or `venv`.

**Step B**

The next step starts with creating a `secrets.toml` file which stores Clarifai's PAT and defines the language learning models that will be available to the chatbot.

This file will hold both the PAT (personal authotization token) for your app, which you would never want to publicly share. The other line is our default models, which isn't an important secret but determines which LLMs you'll offer.

Here's an example `secrets.toml`. Note that when hosting this on the Streamlit cloud, you need to go into your app settings -> secrets to add these lines so that the Streamlit servers can use the information.

```
CLARIFAI_PAT = 'YOUR_PAT_GOES_HERE'
DEFAULT_MODELS = "GPT-3_5-turbo:openai;chat-completion, GPT-4:openai;chat-completion, claude-v2:anthropic;completion, llama2-70b-chat:meta;Llama-2"
```

On Streamlit's cloud, this would appear like this:

![](./assets/streamlit-secrets.jpg)


**Step C**

The second step entails setting up the Streamlit app (`app.py`). I've broken it up into several substeps since this is long section.

1.  Importing Python libraries and modules:
    ```python
    import streamlit as st
    from prompts import instructions_data
    from clarifai_utils.modules.css import ClarifaiStreamlitCSS
    from langchain.llms import OpenAI
    from langchain.agents import AgentType, initialize_agent, load_tools
    from langchain.callbacks import StreamlitCallbackHandler
    from langchain.llms import Clarifai
    from langchain import PromptTemplate, LLMChain
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory, ChatMessageHistory
    from langchain.schema import HumanMessage, AIMessage
    import streamlit.components.v1 as components
    ```
    Import essential APIs and modules needed for the application like Streamlit for app interface, Clarifai for interface with Clarifai API, and Chat related APIs.

2.  **Set the layout:**
    ```python
    st.set_page_config(layout="wide")
    ```
    Configure the layout of the Streamlit app to "wide" layout which allows using more horizontal space on the page.

3.  Define helper functions:
    ```python
    # Load PAT and checks if it exists
    def load_pat():
      if 'CLARIFAI_PAT' not in st.secrets:
        st.error("You need to set the CLARIFAI_PAT in the secrets.")
        st.stop()
      return st.secrets.CLARIFAI_PAT

    # Load models and check if they exist
    def get_default_models():
        # check for DEFAULT_MODELS in secrets.toml and fetch all default models

    # Display previous chat messages and store them into memory
    def show_previous_chats():
        # Rewrite past chats to Streamlit with st.chat_message and save them to memory
    
    # Handles the chat interaction
    def chatbot():
        # Function that handles the interactions between user and AI using Streamlit's chat_input and chat_message features
    ```
    These functions ensure we load the PAT and LLMs, keep a record of chat history, and handle interactions in the chat between the user and the AI.

4.  Define prompt lists and load PAT:
    ```
    prompt_list = list(instructions_data.keys()) 
    pat = load_pat()
    models_map, select_map = get_default_models()
    default_llm = "GPT-4"
    llms_map = {'Select an LLM':None}
    llms_map.update(select_map)
    ```
    Define the list of available prompts along with the personal authentication token (PAT) from the `secrets.toml` file. Select models and append them to the `llms_map`.

5.  Prompt the user for prompt selection:
    ```python
    # Prompt selection
    chosen_instruction_key = st.selectbox(
        'Select a prompt',
        options=prompt_list
        ...)
    ```
    Use Streamlit's built-in select box widget to prompt the user to select one of the provided prompts from `prompt_list`.

6.  Choose the LLM:
    ```python
    if 'chosen_llm' not in st.session_state.keys():
        chosen_llm = st.selectbox(label="Select an LLM", options=llms_map.keys())
        # select and save the LLM chosen by the user
    ```
    Present a choice of language learning models (LLMs) to the user to select the desired LLM.

7.  Initialize the model and set the chatbot instruction:
    ```python
    if "chosen_llm" in st.session_state.keys():
        # load the selected LLM or default LLM if no LLM chose

    # Set the chatbot instruction
    instruction = instructions_data[st.session_state['chosen_instruction_key']]['instruction']
    ```
    Load the language model selected by the user. Initialize the chat with the selected prompt.

8.  Initialize the conversation chain:
    ```python
    # Initialize ConversationChain
    conversation = ConversationChain(
    prompt=prompt,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant", memory_key="chat_history"),
    )
    ```
    Use a `ConversationChain` to handle making conversations between the user and the AI.

9.  Initialize the chatbot:
    ```python
    if "chosen_llm" in st.session_state.keys() and "chat_history" not in st.session_state.keys():
        # Initialize the chatbot and store the first message from the AI 
    ```
    Use the model to generate the first message and store it into the chat history in the session state.

10. Manage Conversation and Display Messages:
    ```python
    if "chosen_llm" in st.session_state.keys():
        show_previous_chats()
        chatbot()
    ```
    Show all previous chats and call `chatbot()` function to continue the conversation.
    
That's the step-by-step walkthrough of what each section in `app.py` does.

