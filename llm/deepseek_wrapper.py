import os
from typing import List

import requests


class DeepSeekWrapper:
    def __init__(self):
        """
        DeepSeek inference interface wrapper, maintaining the same external calling method as QwenWrapper (chat(messages, temperature)).
        It is recommended to configure authentication information via the DEEPSEEK_API_KEY environment variable.
        """
        # Temporarily hardcoded key for quick local testing. Please switch back to environment variable reading before going live or submitting code.
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "") or "sk-d942e96902514cc88890d51e1504ec34"
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List, temperature: float = 0.7) -> str:
        """
        Call DeepSeek Chat Completion interface and return the first candidate response.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        try:
            resp = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0]["message"]["content"]
            return "[No valid response]"
        except Exception as e:
            return f"[Call Exception] {str(e)}"


if __name__ == '__main__':
    ds = DeepSeekWrapper()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who are you?'}
    ]
    response = ds.chat(messages)
    print(response)
