# This script demonstrates how to use LangChainAPIInference with the lmbase framework.
# It loads environment variables (such as API keys) from a .env file, sends a simple
# InferInput to the API, and prints the results including prompt, response, raw_response,
# cost, token usage details, and any extra metadata returned by the inference engine.写pyth

from lmbase.inference.api_call import LangChainAPIInference, InferInput
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    print("Testing LangChainAPIInference with AIHubMix...")
    aihubmix_api_call = LangChainAPIInference(
        lm_provider="aihubmix", lm_name="DeepSeek-V3.2-Exp"
    )
    chatbot = InferInput(
        system_msg="you are a repeater, you should replay what the xiaoming say",
        user_msg="xiaoming said:{xiaoming_words}",
    )
    result = aihubmix_api_call.run(chatbot, xiaoming_words="ni hao")
    print("prompt:", result.prompt)
    print("======")
    print("response:", result.response)
    print("======")
    print("raw_response:", result.raw_response)
    print("======")
    print("cost:", result.cost)

    print("======")

    print("Testing LangChainAPIInference with Doubao...")
    aihubmix_api_call = LangChainAPIInference(
        lm_provider="doubao", lm_name="doubao-seed-1-6-251015"
    )
    chatbot = InferInput(
        system_msg="you are a repeater, you should replay what the xiaoming say",
        user_msg="xiaoming said:{xiaoming_words}",
    )
    result = aihubmix_api_call.run(chatbot, xiaoming_words="ni hao")
    print("prompt:", result.prompt)
    print("======")
    print("response:", result.response)
    print("======")
    print("raw_response:", result.raw_response)
    print("======")
    print("cost:", result.cost)
