import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

import requests
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from termcolor import colored

load_dotenv()

class ContextWindowExceededException(Exception):
    """Exception raised when the context window is exceeded"""
    pass

class Agent:
    """
    LLM Agent supporting Groq, Ollama, LM Studio servers, and AWS Bedrock
    """
    def __init__(
        self,
        model: str,
        systemPrompt: str,
        server: Literal["local", "groq", "lm"]
    ):
        self.model = model
        self.systemPrompt = systemPrompt
        self.server = server
        self.headers = {"Content-Type": "application/json"}
        


    def _handleOllama(self, query: str) -> Optional[str]:
        """Handle Ollama server requests"""
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                headers=self.headers,
                json={
                    "model": self.model,
                    "prompt": query,
                    "system": self.systemPrompt,
                    "stream": False,
                    "temperature": 0,
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.HTTPError as e:
            error_message = str(e.response.text) if hasattr(e, 'response') else str(e)
            if "context window" in error_message.lower():
                raise ContextWindowExceededException("Ollama context window exceeded")
            print(f"Ollama error: {str(e)}")
            return None
        except Exception as e:
            print(f"Ollama error: {str(e)}")
            return None

    def _handleGroq(self, query: str) -> Optional[str]:
        """Handle Groq server requests"""
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.systemPrompt},
                    {"role": "user", "content": query}
                ],
                model=self.model  # Use the model specified in init
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            # Check for context window related errors
            if any(term in error_message.lower() for term in ["context window", "token limit", "too long", "maximum context"]):
                raise ContextWindowExceededException(f"Groq context window exceeded: {error_message}")
            print(f"Groq error: {error_message}")
            return None

    def _handleLMStudio(self, query: str) -> Optional[str]:
        """Handle LM Studio server requests"""
        try:
            client = OpenAI(
                base_url="http://localhost:8001/v1",
                api_key="lm-studio"
            )
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.systemPrompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            if any(term in error_message.lower() for term in ["context window", "token limit", "too long", "maximum context"]):
                raise ContextWindowExceededException(f"LM Studio context window exceeded: {error_message}")
            print(f"LM Studio error: {error_message}")
            return None

    def runAgent(self, query: str) -> Optional[str]:
        """
        Run the agent with the specified query
        
        Args:
            query: Input text for the LLM
            
        Returns:
            Optional[str]: Response from the LLM or None if error occurs
            
        Raises:
            ContextWindowExceededException: If the context window is exceeded
        """
        if not isinstance(query, str) or not query.strip():
            print("Error: Query must be a non-empty string")
            return None

        handlers = {
            "local": self._handleOllama,
            "groq": self._handleGroq,
            "lm": self._handleLMStudio,
        }

        handler = handlers.get(self.server.lower())
        if not handler:
            print(f"Error: Unknown server type '{self.server}'")
            return None

        return handler(query)

