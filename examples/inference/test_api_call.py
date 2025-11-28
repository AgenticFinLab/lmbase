# This script demonstrates how to use LangChainAPIInference with the lmbase framework.
# It loads environment variables (such as API keys) from a .env file, sends a simple
# InferInput to the API, and prints the results including prompt, response, raw_response,
# cost, token usage details, and any extra metadata returned by the inference engine.写pyth

from lmbase.inference.api_call import LangChainAPIInference, InferInput
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    api_call = LangChainAPIInference()

    ii = InferInput(
        system_msg="you are a repeater, you should replay what the use say",
        user_msg="hello world",
    )

    result = api_call.run(ii)
    print("prompt:", result.prompt)
    print("response:", result.response)
    print("raw_response:", result.raw_response)
    print("cost:", result.cost)
    print("prompt_tokens:", result.prompt_tokens)
    print("response_tokens:", result.response_tokens)
    print("raw_response_tokens:", result.raw_response_tokens)
    print("extras:", result.extras)
