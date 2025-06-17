import subprocess
import os

kepler_metrics = {
    "total": "kepler_container_joules_total",
    "package": "kepler_container_package_joules_total",
    "dram": "kepler_container_dram_joules_total",
    "other": "kepler_container_other_joules_total",
    "gpu": "kepler_container_gpu_joules_total"
}


def find_pod(pod_name_start):
    result = subprocess.run(
        ["kubectl", "get", "pods", "--no-headers", "-o", "custom-columns=:metadata.name", "-A"],
        capture_output=True,
        text=True,
        check=True
    )
    pod_names = result.stdout.strip().splitlines()
    for pod_name in pod_names:
        if pod_name.startswith(pod_name_start):
            return pod_name
    return None


def add_postfix_to_filename(file_path, postfix):
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{postfix}{ext}"
    return os.path.join(directory, new_filename)


def construct_prometheus_command(pod_name, metric, start_time, end_time, filename, step="5"):
    if not start_time or not end_time:
        raise ValueError("Start and end time must be specified")
    url = f"http://localhost:9090/api/v1/query_range?query={metric}&start={start_time}&end={end_time}&step={step}"
    return f"kubectl exec -n monitoring {pod_name} -- wget -qO- '{url}' > {filename}"


def collect_energy(pod_name_start='prometheus-prometheus-kube-prometheus-prometheus', start_time=None, end_time=None,
                   filename=None, step="5"):
    if filename is None:
        raise ValueError("Filename must be specified")

    pod_name = find_pod(pod_name_start)
    if pod_name is None:
        raise ValueError(f"Could not find pod with name starting with {pod_name_start}")

    for postfix, metric in kepler_metrics.items():
        output_filename = add_postfix_to_filename(filename, postfix)
        command = construct_prometheus_command(pod_name, metric, start_time, end_time, output_filename, step)
        print(f"Executing command for {postfix}: {command}")
        subprocess.run(command, shell=True, check=True)
