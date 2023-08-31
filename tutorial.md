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

