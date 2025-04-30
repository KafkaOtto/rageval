import bz2
import json
from loguru import logger


def initialize_batch():
    """ Helper function to create file objects. """
    return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}


def load_data(dataset_path):
    try:
        cnt = 0
        with bz2.open(dataset_path, "rt") as file:
            data = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    for key in data:
                        data[key].append(item[key])
                    cnt += 1
                    if cnt == 2:
                        break
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            return data
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e

def load_data_in_batches(dataset_path: str,
                         batch_size: int
                         ):
    try:
        print(f"batch_size {batch_size}")
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    print(f"item: {item.keys()}")
                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e

# if __name__ == '__main__':
#     dataset_path = "../../dataset/crag_task_1_and_2_dev_v4.jsonl.bz2"
#     data = load_data(dataset_path)
#     logger.info(f"get data size: {len(data["query"])}")
if __name__ == '__main__':
    from tqdm import tqdm
    dataset_path = "../dataset/crag_task_1_and_2_dev_v4.jsonl.bz2"
    batch_size = 2
    iter = 1;
    total_size = 0
    for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc="Loading Batches"):
        print(f"batch size: {len(batch['query'])}")
        total_size += len(batch['query'])
        iter += 1
    print(f"Total size: {total_size}")
    print(f"iter: {iter}")