"""
Test Script for LLM API Call Functions

This module provides basic testing for the LLM client builder functions
located in lmbase/inference/api_call.py. It tests both AIHubMix integration
and multi-provider LLM client construction with simple test cases.

"""

import os
import sys
from dotenv import load_dotenv
from lmbase.inference.api_call import build_llm_aihubmix, build_llm


def test_aihubmix_completion_basic():
    """
    Test basic AIHubMix completion functionality with simple prompt.
    """
    print("Testing AIHubMix Basic Completion")

    try:
        response = build_llm_aihubmix(
            model_id="DeepSeek-V3.2-Exp",
            prompt="请介绍一下蓝天格锐案件",
            temperature=0.1,
            max_tokens=5000,
        )

        # Basic validation
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        print("Basic completion test passed")
        print(f"Response: {response}")
        return True

    except Exception as e:
        print(f"Basic completion test failed: {e}")
        return False


def test_aihubmix_client_only():
    """
    Test AIHubMix client-only configuration without execution.
    """
    print("Testing AIHubMix Client-Only Mode")

    try:
        client = build_llm_aihubmix(
            model_id="DeepSeek-V3.2-Exp", return_client_only=True
        )

        # Validate client object
        assert client is not None
        assert hasattr(client, "chat")
        assert hasattr(client.chat.completions, "create")

        print("Client-only test passed")
        return True

    except Exception as e:
        print(f"Client-only test failed: {e}")
        return False


def test_build_llm_different_providers():
    """
    Test build_llm function with different provider configurations.
    """
    print("Testing build_llm with different providers")

    test_cases = [
        # (model_name, description)
        ("gpt-3.5-turbo", "OpenAI model"),
        ("deepseek-coder", "DeepSeek model"),
        ("qwen-plus", "Qwen model"),
        ("doubao-7b", "Doubao model"),
    ]

    passed_tests = 0

    for model_name, description in test_cases:
        try:
            # Skip if required API keys are not available
            if "doubao" in model_name.lower() and not os.getenv("DOUBAO_API_KEY"):
                print(f"Skipping {description} - DOUBAO_API_KEY not set")
                continue
            elif "deepseek" in model_name.lower() and not os.getenv("DEEPSEEK_API_KEY"):
                print(f"Skipping {description} - DEEPSEEK_API_KEY not set")
                continue
            elif "gpt" in model_name.lower() and not os.getenv("OPENAI_API_KEY"):
                print(f"Skipping {description} - OPENAI_API_KEY not set")
                continue
            elif "qwen" in model_name.lower() and not os.getenv("DASHSCOPE_API_KEY"):
                print(f"Skipping {description} - DASHSCOPE_API_KEY not set")
                continue

            # Test client creation
            llm_client = build_llm(model_override=model_name, temperature=0.1)

            assert llm_client is not None
            assert hasattr(llm_client, "invoke")

            print(f"{description} test passed")
            passed_tests += 1

        except Exception as e:
            print(f"{description} test failed: {e}")

    return passed_tests > 0


def main():
    """
    Main test execution function.
    Runs all test cases and reports overall results.
    """
    print("Starting LLM API Call Tests")
    print("=" * 50)

    # Load environment variables for API keys
    load_dotenv()

    # Run test cases
    test_results = []

    test_results.append(test_aihubmix_completion_basic())
    test_results.append(test_aihubmix_client_only())
    # test_results.append(test_build_llm_different_providers())

    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(test_results)
    total = len(test_results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("All tests passed successfully")
    else:
        print("Some tests failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
