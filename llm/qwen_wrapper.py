import dashscope  # Alibaba Cloud official SDK, encapsulates HTTP details for Qwen API requests
from typing import List  # Data type: list

class QwenWrapper:
    def __init__(self):  # Initialize Qwen API, set API key, Base URL, model and request headers
        self.api_key = "sk-bcabe4992cb94e8f896126cef8ee8dea"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
        self.model = "qwen2.5-72b-instruct"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List, temperature: float = 0.7) -> str:  # Pass message list and temperature, call Qwen API to generate response
        """
        Call Qwen API to generate response.

        messages: List of dicts, e.g. [{"role": "user", "content": "Who are you?"}]
        Each element is a dictionary containing 'role' and 'content' keys
        """
        try:  # Call Qwen API to generate response, return error message if failed
            resp = dashscope.Generation.call(  # Call Qwen API
                api_key=self.api_key,  # API key
                model=self.model,  # Model
                messages=messages,  # Message list
                result_format="message",  # Result format
                temperature=temperature,  # Temperature
                top_p = 0.8  # Top-p sampling
            )
            if (
                    hasattr(resp, "output")  # If response has output attribute
                    and "choices" in resp.output  # And output has choices key
                    and resp.output["choices"]  # And choices value is not empty
            ):
                return resp.output["choices"][0]["message"]["content"]  # Return response content
            else:
                return "[No valid response]"  # Otherwise return "No valid response"
        except Exception as e:
            return f"[API call exception] {str(e)}"  # If Qwen API call fails, return call exception and error message


if __name__ == '__main__':  # Test Qwen API, pass message list, print response content
    qwen = QwenWrapper()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Who are you?'}
    ]
    response = qwen.chat(messages)
    print(response)

"""
Example output:

I am Qwen, a large-scale language model developed by Alibaba Cloud. I can generate various types of text, such as articles, stories, poems, etc., and can adjust and optimize according to different scenarios and needs. In addition, I also have code writing capabilities and can help solve programming problems. If you have any questions or need help, please feel free to ask me!
"""
