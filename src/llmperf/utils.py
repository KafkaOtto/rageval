import json
import time
from typing import Any, Dict



RESULTS_VERSION = "2023-08-31"


class LLMPerfResults:
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata or {}
        self.timestamp = int(time.time())
        self.metadata["timestamp"] = self.timestamp
        self.version = RESULTS_VERSION

    def to_dict(self):
        data = {
            "version": self.version,
            "name": self.name,
        }
        data.update(self.metadata)
        data = flatten_dict(data)
        return data

    def json(self):
        data = self.to_dict()
        return json.dumps(data)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_answer_score_mapping(response_file: str, accuracy_file: str):
    answer_score_mapping = {}

    with open(response_file, "r") as response_f:
        responses = json.load(response_f)

    with open(accuracy_file, "r") as accuracy_f:
        accuracies = json.load(accuracy_f)

    print("query size", len(responses))

    for response, accuracy in zip(responses, accuracies):
        answer = response["generated_text"]
        score = accuracy["score"]
        question = response["request_config"]["prompt"][0]
        explanation = accuracy["explanation"]

        answer_lowercase = str(answer).strip().rstrip('.').lower()

        if question not in answer_score_mapping:
            answer_score_mapping[question] = []

        answer_score_mapping[question].append({
            "score": score,
            "answer": answer_lowercase,
            "explanation": explanation,
            "request_config": accuracy["request_config"],
        })
    return answer_score_mapping


def check_existing_score(query, prediction, answer_score):

    mapping_value = answer_score.get(query)
    if mapping_value:
        prediction_normalized = str(prediction).strip().rstrip('.').lower()
        for value in mapping_value:
            if value["answer"] == prediction_normalized:
                return value["explanation"], value["score"], value["request_config"]
    return None

if __name__ == '__main__':
    answer_score_mapping = {}
    check_existing_score("a", "b", answer_score_mapping)
