import argparse
import json
import os
from pathlib import Path
import time
import random
from typing import Any, Dict, List, Optional, Tuple
import ray

from llmperf import common_metrics
from llmperf.common import construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    LLMPerfResults,
)
from llmperf.energy_collection import collect_energy
from tqdm import tqdm

from llmperf.data_loader import load_data

def load_raw_results(complete_requests):
    raw_results = []
    num_errored_requests = 0
    num_mismatched_requests = 0
    for out in complete_requests:
        metrics, generated_text, completed_request_config = out
        raw_results.append(
            {
                "metrics": metrics,
                "generated_text": generated_text,
                "request_config": dict(completed_request_config),
            }
        )
    return raw_results, num_errored_requests, num_mismatched_requests

def get_accuracies_latencies(
        model: str,
        additional_sampling_params: Optional[Dict[str, Any]] = None,
        num_concurrent_requests: int = 1,
        test_timeout_s=90,
        llm_api="RAG",
        dataset: Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)

    if not additional_sampling_params:
        additional_sampling_params = {}

    num_completed_requests = 0
    # make up prompts outside of send loop for faster benchmarking loop

    prompts = dataset["query"]
    answers = dataset["answer"]
    max_num_completed_requests = len(prompts)

    start_time = time.time()
    pbar = tqdm(total=max_num_completed_requests)

    clients = construct_clients(llm_api=llm_api, num_clients=num_concurrent_requests)
    req_launcher = RequestsLauncher(clients)
    iter = 0
    completed_requests = []

    while (
            time.time() - start_time < test_timeout_s
            and num_completed_requests < max_num_completed_requests
    ):
        prompt = prompts[iter]
        request_config = RequestConfig(
            model=model,
            prompt=(prompt, len(prompt)),
            sampling_params=additional_sampling_params,
            llm_api=llm_api,
            metadata={"answer": answers[iter]},
        )
        req_launcher.launch_requests(request_config)
        iter += 1
        if not (iter % num_concurrent_requests) or iter == max_num_completed_requests:
            outs = req_launcher.get_next_ready()
            completed_requests.extend(outs)
        pbar.update(len(completed_requests) - num_completed_requests)
        num_completed_requests = len(completed_requests)
    pbar.close()
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    if total_elapsed_time>= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")

    # print(f"completed requests: {completed_requests}")
    raw_results, num_errored_requests, num_mismatched_requests = load_raw_results(completed_requests)
    summary_metrics = {}
    summary_metrics["start_time"] = start_time
    summary_metrics["end_time"] = end_time
    summary_metrics[common_metrics.NUM_ERRORS] = num_errored_requests
    summary_metrics["num_mismatched_requests"] = num_mismatched_requests
    summary_metrics["error_rate"] = num_errored_requests / num_completed_requests
    summary_metrics["mismatch_rate"] = num_mismatched_requests / num_completed_requests
    summary_metrics[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    summary_metrics["num_non_errored_requests"] = num_completed_requests - num_errored_requests

    # Metadata
    summary_metrics["model"] = model
    summary_metrics["num_concurrent_requests"] = num_concurrent_requests
    summary_metrics["additional_sampling_params"] = additional_sampling_params
    summary_metrics["llm_api"] = llm_api

    return summary_metrics, raw_results

def run_job(model: str,
        filename: str,
        dataset: Dict[str, Any] = None,
        additional_sampling_params: Optional[Dict[str, Any]] = None,
        num_concurrent_requests: int = 1,
        test_timeout_s=90,
        llm_api="RAG",
        output_dir: str = None,
        rerun: bool = False,
              ):
    # precheck if results already exist
    summary_file_name = f"{filename}_summary"
    responses_file_name = f"{filename}_responses"
    energy_file_name = f"{filename}_energy"
    if not rerun and check_results_exist_and_pass(output_dir, summary_file_name, responses_file_name):
        print(f"current batch: {filename} is successful")
        return

    summary_metrics, raw_results = get_accuracies_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
        dataset=dataset,
    )

    # save results
    print("Warmup Summary metrics:")
    print(summary_metrics)
    # print("Warmup Raw results:")
    # print(raw_results)
    save_results(output_dir, energy_file_name, summary_file_name, responses_file_name, raw_results, summary_metrics)

def run_batch(model: str,
        warmup_input_dir: str,
        prod_input_dir: str,
        additional_sampling_params: Optional[Dict[str, Any]] = None,
        num_concurrent_requests: int = 1,
        test_timeout_s=90,
        llm_api="RAG",
        treatment_id: int = 1,
        output_dir: str = None,
        rerun: bool = False,
              ):
    warmup_data = load_data(warmup_input_dir)
    prod_data = load_data(prod_input_dir)
    # precheck if results already exist
    warmup_filename = f"{model}_batch_{treatment_id}_warmup"
    prod_filename = f"{model}_batch_{treatment_id}_prod"
    run_job(model=model, filename=warmup_filename, dataset=warmup_data,
            additional_sampling_params=additional_sampling_params,
            num_concurrent_requests=num_concurrent_requests,
            test_timeout_s=test_timeout_s,
            llm_api=llm_api,
            output_dir=output_dir,
            rerun=rerun)
    run_job(model=model, filename=prod_filename, dataset=prod_data,
            additional_sampling_params=additional_sampling_params,
            num_concurrent_requests=num_concurrent_requests,
            test_timeout_s=test_timeout_s,
            llm_api=llm_api,
            output_dir=output_dir,
            rerun=rerun)

def check_results_exist_and_pass(output_dir,
                 summary_filename,
                 responses_filename):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return False
    elif not output_dir.is_dir():
        raise ValueError(f"{output_dir} is not a directory")
    summary_file = output_dir / f"{summary_filename}.json"
    if not summary_file.exists():
        return False
    responses_file = output_dir / f"{responses_filename}.json"
    if not responses_file.exists():
        return False
    # read from summary file and check if number_error is bigger than 0. If so return False. Otherwise, return true
    try:
        with open(summary_file, "r", encoding="utf-8") as file:
            summary_data = json.load(file)

        if summary_data.get("number_errors", 0) > 0:
            return False
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Error reading summary file {summary_file}: {e}")

    return True

def save_results(output_dir,
                 energy_file_name,
                 summary_filename,
                 responses_filename,
                 response_results,
                 summary_metrics):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir():
        raise ValueError(f"{output_dir} is not a directory")
    save_summary_results(output_dir, summary_filename, summary_metrics)

    try:
        with open(output_dir / f"{responses_filename}.json", "w") as f:
            json.dump(response_results, f, indent=4)
    except Exception as e:
        print(response_results)
        raise e
    save_energy_results(output_dir,
                 energy_filename=energy_file_name,
                 summary_metrics=summary_metrics)

def save_energy_results(output_dir,
                 energy_filename,
                 summary_metrics):
    full_path = output_dir / f"{energy_filename}.json"
    start_time = summary_metrics["start_time"]
    end_time = summary_metrics["end_time"]
    collect_energy(start_time = start_time,
                   end_time = end_time,
                   filename = full_path)

def save_summary_results(output_dir,
                 summary_filename,
                 summary_metrics):
    summary_results = LLMPerfResults(name=summary_filename, metadata=summary_metrics)
    try:
        with open(output_dir / f"{summary_filename}.json", "w") as f:
            json.dump(summary_results.to_dict(), f, indent=4, default=str)
    except Exception as e:
        print(summary_results.to_dict())
        raise e

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})

    parser = argparse.ArgumentParser(description="Run a token throughput and latency benchmark.")

    # Define arguments
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")
    parser.add_argument("--model", type=str, help="The model to use for this load test.")
    parser.add_argument("--num-concurrent-requests", type=int, help="The number of concurrent requests to send.")
    parser.add_argument("--timeout", type=int, help="The amount of time to run the load test for.")
    parser.add_argument("--warmup-input-dir", type=str, help="The directory to read the dataset from.")
    parser.add_argument("--prod-input-dir", type=str, help="The directory to read the dataset from.")
    parser.add_argument("--output-dir", type=str, help="The directory to save the results to.")
    parser.add_argument("--llm-api", type=str, help="The name of the LLM API to use.")
    parser.add_argument("--metadata", type=str, help="Metadata for the test, e.g. name=benchmark,version=1")
    parser.add_argument("--treatment-id", type=str, help="For output filename id")
    parser.add_argument("--rerun", type=bool, help="If rerun the experiment")
    parser.add_argument(
        "--additional-sampling-params", type=str, default="{}", help=(
            "Additional sampling params to send with the each request to the LLM API. "
            "(default: %(default)s) No additional sampling params are sent."
        ),
    )
    args = parser.parse_args()
    if args.config:
        config_data = load_config(args.config)
        parser.set_defaults(**config_data)
    args = parser.parse_args()
    run_batch(
        llm_api=args.llm_api,
        model=args.model,
        test_timeout_s=args.timeout,
        num_concurrent_requests=args.num_concurrent_requests,
        treatment_id=args.treatment_id,
        additional_sampling_params=args.additional_sampling_params,
        warmup_input_dir=args.warmup_input_dir,
        prod_input_dir=args.prod_input_dir,
        output_dir=args.output_dir,
        rerun=args.rerun
    )
