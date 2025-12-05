import os
from typing import List

import requests


class DeepSeekWrapper:
    def __init__(self):
        """
        DeepSeek inference API wrapper, maintains the same external interface as QwenWrapper (chat(messages, temperature)).
        It is recommended to configure authentication via environment variable DEEPSEEK_API_KEY.
        """
        # ⚠️ Temporarily hardcoded key for local quick testing. Change back to environment variable reading before deployment or code submission.
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "") or "sk-d942e96902514cc88890d51e1504ec34"
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-chat"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List, temperature: float = 0.7) -> str:
        """
        Call DeepSeek Chat Completion API and return the first candidate response.
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
            return f"[API call exception] {str(e)}"


if __name__ == '__main__':
    ds = DeepSeekWrapper()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who are you?'}
    ]
    response = ds.chat(messages)
    print(response)
