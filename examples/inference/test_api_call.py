"""
Comprehensive Test Examples for LLM API Calls

This module demonstrates various usage patterns for the LLM client builders
and base classes. It shows how to use different providers, request formats,
and response handling strategies.

Before running these examples, ensure you have set the required environment
variables for your chosen providers.
"""

import os
from dotenv import load_dotenv
from lmbase.inference.api_call import build_llm_aihubmix, build_llm
from lmbase.inference.base import (
    LLMProvider,
    Message,
    LLMRequest,
    LLMResponse,
    detect_provider,
)


def example_1_basic_aihubmix_usage():
    """
    Example 1: Basic AIHubMix usage with simple prompt.

    Demonstrates the most straightforward way to use AIHubMix with
    a simple text prompt and default parameters.
    """
    print("=== Example 1: Basic AIHubMix Usage ===")

    try:
        # Simple one-line call with just model and prompt
        response = build_llm_aihubmix(
            model_id="gpt-3.5-turbo",  # Model from AIHubMix catalog
            prompt="Explain the concept of machine learning in one sentence.",
        )

        print(f"Response: {response}")
        print()

    except Exception as e:
        print(f"Error in example 1: {e}")
        print(
            "Make sure AIHUBMIX_API_KEY and AIHUBMIX_BASE_URL are set in your .env file"
        )
        print()


def example_2_advanced_aihubmix_with_messages():
    """
    Example 2: Advanced AIHubMix usage with structured messages.

    Shows how to use the messages parameter for multi-turn conversations
    and system prompts with custom generation parameters.
    """
    print("=== Example 2: Advanced AIHubMix with Structured Messages ===")

    try:
        # Structured conversation with system message and custom parameters
        messages = [
            {
                "role": "system",
                "content": "You are a helpful math tutor. Explain concepts clearly and provide examples.",
            },
            {"role": "user", "content": "What is the Pythagorean theorem?"},
        ]

        response = build_llm_aihubmix(
            model_id="gpt-4",
            messages=messages,
            temperature=0.3,  # Lower temperature for more deterministic output
            max_tokens=500,  # Limit response length
            top_p=0.9,  # Use nucleus sampling
            seed=42,  # Fixed seed for reproducible results
        )

        print(f"Math Tutor Response: {response}")
        print()

    except Exception as e:
        print(f"Error in example 2: {e}")
        print()


def example_3_aihubmix_client_reuse():
    """
    Example 3: AIHubMix client reuse for multiple requests.

    Demonstrates how to get a client instance and reuse it for
    multiple API calls, which is more efficient for batch processing.
    """
    print("=== Example 3: AIHubMix Client Reuse ===")

    try:
        # Get the client instance
        client = build_llm_aihubmix("gpt-3.5-turbo", return_client_only=True)

        # Prepare multiple requests using the base classes
        requests = [
            LLMRequest.from_simple_prompt(
                model="gpt-3.5-turbo",
                prompt="What is the capital of France?",
                temperature=0.1,
            ),
            LLMRequest.from_simple_prompt(
                model="gpt-3.5-turbo",
                prompt="Explain quantum computing briefly.",
                temperature=0.5,
            ),
        ]

        # Execute multiple requests with the same client
        for i, request in enumerate(requests, 1):
            response = client.call(request)
            print(f"Response {i}: {response.content}")
            print(f"Tokens used: {response.total_tokens}")
            print()

    except Exception as e:
        print(f"Error in example 3: {e}")
        print()


def example_4_langchain_integration():
    """
    Example 4: LangChain integration with multiple providers.

    Shows how to use the build_llm function to create LangChain
    ChatOpenAI clients for different providers.
    """
    print("=== Example 4: LangChain Integration ===")

    # Test different providers (comment out ones you don't have API keys for)
    providers_to_test = [
        # ("doubao-model-name", "Doubao"),  # Uncomment if you have Doubao API key
        # ("deepseek-coder", "DeepSeek"),   # Uncomment if you have DeepSeek API key
        ("gpt-3.5-turbo", "OpenAI"),  # Requires OPENAI_API_KEY
        # ("qwen-plus", "Qwen")             # Uncomment if you have DASHSCOPE_API_KEY
    ]

    for model_name, provider_name in providers_to_test:
        try:
            print(f"Testing {provider_name} with model {model_name}:")

            # Build the LangChain client
            llm = build_llm(model_override=model_name, temperature=0.1)

            # Simple test call
            response = llm.invoke("Say 'Hello World' in a creative way.")
            print(f"Response: {response.content}")
            print()

        except Exception as e:
            print(f"Error with {provider_name}: {e}")
            print("Make sure the required API key is set in your environment")
            print()


def example_5_comprehensive_base_classes():
    """
    Example 5: Comprehensive base class usage.

    Demonstrates the full power of the base classes for creating
    complex requests and handling detailed responses.
    """
    print("=== Example 5: Comprehensive Base Class Usage ===")

    try:
        # Create a complex conversation with multiple messages
        messages = [
            Message(role="system", content="You are a knowledgeable historian."),
            Message(role="user", content="Tell me about ancient Rome."),
            Message(
                role="assistant", content="Ancient Rome was a powerful civilization..."
            ),
            Message(
                role="user",
                content="What were their major contributions to engineering?",
            ),
        ]

        # Create a detailed request
        request = LLMRequest(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop=["\n\n", "###"],  # Stop sequences
            seed=12345,
        )

        # Get client and execute request
        client = build_llm_aihubmix("gpt-3.5-turbo", return_client_only=True)
        response = client.call(request)

        # Demonstrate response properties
        print(f"Content: {response.content}")
        print(f"Model: {response.model}")
        print(f"Finish Reason: {response.finish_reason}")
        print(f"Token Usage: {response.usage}")
        print(f"Prompt Tokens: {response.prompt_tokens}")
        print(f"Completion Tokens: {response.completion_tokens}")
        print(f"Total Tokens: {response.total_tokens}")
        print()

        # Convert to dictionary
        response_dict = response.to_dict()
        print(f"Response as dict: {list(response_dict.keys())}")
        print()

    except Exception as e:
        print(f"Error in example 5: {e}")
        print()


def example_6_provider_detection():
    """
    Example 6: Provider detection functionality.

    Shows how to use the provider detection utilities to
    automatically determine the provider from model names.
    """
    print("=== Example 6: Provider Detection ===")

    test_models = [
        "doubao-7b-chat",
        "deepseek-coder-33b-instruct",
        "gpt-4-turbo-preview",
        "qwen-plus",
        "aihubmix-special-model",
        "unknown-model-v1",
    ]

    for model in test_models:
        provider = detect_provider(model)
        print(f"Model: {model:30} -> Provider: {provider.value}")

    print()


def example_7_error_handling():
    """
    Example 7: Comprehensive error handling.

    Demonstrates proper error handling patterns for different
    types of failures that can occur with LLM API calls.
    """
    print("=== Example 7: Error Handling ===")

    # Test 1: Missing required parameters
    try:
        response = build_llm_aihubmix("gpt-3.5-turbo")  # Missing prompt/messages
    except ValueError as e:
        print(f"Expected error caught: {e}")

    # Test 2: Invalid parameter values
    try:
        response = build_llm_aihubmix(
            model_id="gpt-3.5-turbo",
            prompt="Test",
            temperature=2.5,  # Invalid temperature
        )
    except Exception as e:
        print(f"Parameter validation error: {e}")

    print("Error handling examples completed.")
    print()


if __name__ == "__main__":
    """
    Main execution block that runs all examples.

    Load environment variables and execute each example
    sequentially to demonstrate the full capabilities of
    the LLM inference system.
    """
    # Load environment variables from .env file
    load_dotenv()

    print("LLM Inference API Call Examples")
    print("=" * 50)
    print()

    # Run all examples
    example_1_basic_aihubmix_usage()
    example_2_advanced_aihubmix_with_messages()
    example_3_aihubmix_client_reuse()
    # example_4_langchain_integration()
    example_5_comprehensive_base_classes()
    example_6_provider_detection()
    example_7_error_handling()

    print("All examples completed!")
