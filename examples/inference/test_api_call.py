
from lmbase.inference.api_call import LangChainAPIInference,InferInput

if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv()
    api_call = LangChainAPIInference()
    
    ii = InferInput(
        system_msg="you are a repeater, you should replay what the use say", user_msg='hello world'
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