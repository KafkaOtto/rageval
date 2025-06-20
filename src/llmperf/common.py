from typing import List
from llmperf.ray_clients.rag_client import RAGClient
from llmperf.ray_llm_client import LLMClient
from llmperf.ray_clients.openai_accuracy_client import OpenaiAccuracyClient


SUPPORTED_APIS = ["openai", "anthropic", "litellm"]


def construct_clients(llm_api: str, num_clients: int) -> List[LLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    print(f"Constructing {llm_api} clients...")
    if llm_api == "RAG":
        clients = [RAGClient.remote() for _ in range(num_clients)]
    elif llm_api == "openai_acc":
        clients = [OpenaiAccuracyClient.remote() for _ in range(num_clients)]
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return clients
