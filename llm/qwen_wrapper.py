import dashscope        # Alibaba Cloud official SDK, encapsulating HTTP details for requesting Qwen API
from typing import List        # Data type, List

class QwenWrapper:
    def __init__(self):          # Initialize Qwen interface, set API key, Base URL, model, and headers
        self.api_key = "sk-bcabe4992cb94e8f896126cef8ee8dea"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
        self.model = "qwen2.5-72b-instruct"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List, temperature: float = 0.7) -> str:   # Pass message list and temperature, call Qwen API to generate response
        """
        Call Qwen interface to generate response.

        messages: List of dicts, e.g. [{"role": "user", "content": "Who are you?"}]
        The message is a list, where each element is a dictionary containing two keys: role and content.
        """
        try:                                # Call Qwen API to generate response, return error message if failed
            resp = dashscope.Generation.call(         # Call Qwen API
                api_key=self.api_key,                 # API Key
                model=self.model,                     # Model
                messages=messages,                    # List of messages
                result_format="message",              # Result format
                temperature=temperature,              # Temperature
                top_p = 0.8                           # Top P sampling
            )
            if (
                    hasattr(resp, "output")           # If resp has output attribute
                    and "choices" in resp.output      # And choices key exists in output
                    and resp.output["choices"]        # And the value of choices is not empty
            ):
                return resp.output["choices"][0]["message"]["content"]    # Return response content
            else:
                return "[No valid response]"                                    # Otherwise return "[No valid response]"
        except Exception as e:
            return f"[Call Exception] {str(e)}"           # If calling Qwen fails, return call exception and error message


if __name__ == '__main__':           # Test Qwen interface, pass message list, print response content
    qwen = QwenWrapper()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who are you?'}
    ]
    response = qwen.chat(messages)
    print(response)

"""
Example output:

I am Qwen, a large-scale language model developed by Alibaba Cloud. I can generate various types of text, such as articles, stories, poems, etc., and can adjust and optimize according to different scenarios and needs. In addition, I also have coding capabilities and can help solve programming problems. If you have any questions or need help, feel free to ask me at any time!
"""
