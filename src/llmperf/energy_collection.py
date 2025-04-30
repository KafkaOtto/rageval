import subprocess

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

def construct_prometheus_command(pod_name, start_time, end_time, filename, step="1s"):
    if start_time is None:
        raise ValueError(f"Start time must be specified")
    if end_time is None:
        raise ValueError(f"End time must be specified")
    return f"kubectl exec -n monitoring {pod_name} -- wget -qO- 'http://localhost:9090/api/v1/query_range?query=kepler_container_joules_total&start={start_time}&end={end_time}&step=1' > {filename}"

def collect_energy(pod_name_start = 'prometheus-prometheus-kube-prometheus-prometheus', start_time = None, end_time = None, filename = None, step="1s"):
    if filename is None:
        raise ValueError(f"Filename must be specified")
    pod_name = find_pod(pod_name_start)
    if pod_name is None:
        raise ValueError(f"Could not find pod with name starting with {pod_name_start}")
    command = construct_prometheus_command(pod_name, start_time, end_time, filename, step)
    print(f"Executing command: {command}")
    subprocess.run(command, shell=True, check=True)