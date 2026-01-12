import os
import subprocess
import threading
import time


def read_stream(stream, process_type, identifier, stream_name):
    for line in iter(stream.readline, ""):
        print(f"[{process_type} {identifier} - {stream_name}] {line.strip()}")


def read_output(process, process_type, identifier):
    threads = []
    for stream_name, stream in [("STDOUT", process.stdout), ("STDERR", process.stderr)]:
        thread = threading.Thread(
            target=read_stream, args=(stream, process_type, identifier, stream_name)
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    process.stdout.close()
    process.stderr.close()


def run_simulation(server_args, client_args_list):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # 파이썬 출력 버퍼링 해제

    print("Starting server with arguments:", server_args)
    server_process = subprocess.Popen(
        server_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    server_thread = threading.Thread(
        target=read_output, args=(server_process, "Server", "")
    )
    server_thread.start()

    time.sleep(2)  # 서버가 시작될 시간을 확보

    client_processes = []
    client_threads = []

    print("Starting clients with arguments:")
    for i, client_args in enumerate(client_args_list):
        env["CUDA_VISIBLE_DEVICES"] = str(i % 4)

        client_process = subprocess.Popen(
            client_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        client_thread = threading.Thread(
            target=read_output, args=(client_process, "Client", i)
        )
        client_thread.start()

        client_processes.append(client_process)
        client_threads.append(client_thread)

    print("Simulation running. Monitoring processes...")
    try:
        while True:
            server_running = server_process.poll() is None
            clients_running = any(p.poll() is None for p in client_processes)

            if not server_running and not clients_running:
                print("All processes have terminated.")
                break

            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Stopping all processes...")
        if server_process.poll() is None:
            print("Terminating server process")
            server_process.terminate()
            server_process.wait()
            print("Server process has been terminated.")

        for i, client_process in enumerate(client_processes):
            if client_process.poll() is None:
                print(f"Terminating client process {i}")
                client_process.terminate()
                client_process.wait()
                print(f"Client process {i} has been terminated.")

        server_thread.join()
        for t in client_threads:
            t.join()


if __name__ == "__main__":
    common_args = {
        "dataset": "mnli",
        "model": "distilbert",
        "local_epochs": "5",
        "global_epochs": "100",
        "device": "cpu",
        "total_clients": "20",
        "clients_per_round": "5",
        "learning_rate": "0.01",
        "batch_size": "256",
        "port": "3000",
        "optimizer": "sgd",
        "networking_fairness": "False",
        "server_model_aggregation": "True",
    }

    workloads = [
        {
            **common_args,
            "method": "sfl",
            "distributer": "uniform",
            "server_model_aggregation": "True",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.1",
        },
        {
            **common_args,
            "method": "sfl",
            "distributer": "label",
            "labels_per_client": "2",
            "server_model_aggregation": "True",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.1",
        },
        {
            **common_args,
            "method": "sfl",
            "distributer": "dirichlet",
            "dirichlet_alpha": "0.5",
            "server_model_aggregation": "True",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.1",
        },
        {
            **common_args,
            "method": "scala",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.1",
            "distributer": "uniform",
            "batch_size": "1280",
        },
        {
            **common_args,
            "method": "scala",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.1",
            "distributer": "label",
            "labels_per_client": "2",
            "batch_size": "1280",
        },
        {
            **common_args,
            "method": "scala",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.1",
            "distributer": "dirichlet",
            "dirichlet_alpha": "0.5",
            "batch_size": "1280",
        },
        {
            **common_args,
            "method": "scala",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.3",
            "distributer": "dirichlet",
            "dirichlet_alpha": "0.5",
            "batch_size": "1280",
        },
        {
            **common_args,
            "method": "scala",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.5",
            "distributer": "dirichlet",
            "dirichlet_alpha": "0.5",
            "batch_size": "1280",
        },
        {
            **common_args,
            "method": "scala",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.7",
            "distributer": "dirichlet",
            "dirichlet_alpha": "0.5",
            "batch_size": "1280",
        },
        {
            **common_args,
            "method": "scala",
            "split_strategy": "ratio_layer",
            "split_ratio": "0.9",
            "distributer": "dirichlet",
            "dirichlet_alpha": "0.5",
            "batch_size": "1280",
        },
    ]

    for workload in workloads:
        DATASET = workload.get("dataset", None)
        MODEL = workload.get("model", None)
        METHOD = workload.get("method", None)
        LOCAL_EPOCHS = workload.get("local_epochs", None)
        GLOBAL_EPOCHS = workload.get("global_epochs", None)
        DEVICE = workload.get("device", None)
        DISTRIBUTER = workload.get("distributer", None)
        SPLIT_STRATEGY = workload.get("split_strategy", None)
        SPLIT_RATIO = workload.get("split_ratio", None)
        LABELS_PER_CLIENT = workload.get("labels_per_client", None)
        TOTAL_CLIENTS = workload.get("total_clients", None)
        CLIENTS_PER_ROUND = workload.get("clients_per_round", None)
        LEARNING_RATE = workload.get("learning_rate", None)
        BATCH_SIZE = workload.get("batch_size", None)
        DIRICHLET_ALPHA = workload.get("dirichlet_alpha", None)
        PORT = workload.get("port", "1000")
        SERVER_MODEL_AGGREGATION = workload.get("server_model_aggregation", None)
        OPTIMIZER = workload.get("optimizer", None)
        DELETE_FRACTION_OF_DATA = workload.get("delete_fraction_of_data", None)
        NETWORKING_FAIRNESS = workload.get("networking_fairness", None)

        server_command = ["python3", "server/main.py"]

        if DATASET:
            server_command.extend(["-d", DATASET])
        if MODEL:
            server_command.extend(["-m", MODEL])
        if METHOD:
            server_command.extend(["-M", METHOD])
        if LOCAL_EPOCHS:
            server_command.extend(["-le", LOCAL_EPOCHS])
        if GLOBAL_EPOCHS:
            server_command.extend(["-gr", GLOBAL_EPOCHS])
        if DEVICE:
            server_command.extend(["-de", DEVICE])
        if TOTAL_CLIENTS:
            server_command.extend(["-nc", TOTAL_CLIENTS])
        if CLIENTS_PER_ROUND:
            server_command.extend(["-ncpr", CLIENTS_PER_ROUND])
        if DISTRIBUTER:
            server_command.extend(["-distr", DISTRIBUTER])
        if LABELS_PER_CLIENT:
            server_command.extend(["-lpc", LABELS_PER_CLIENT])
        if SPLIT_STRATEGY:
            server_command.extend(["-ss", SPLIT_STRATEGY])
        if SPLIT_RATIO:
            server_command.extend(["-sr", SPLIT_RATIO])
        if LEARNING_RATE:
            server_command.extend(["-lr", LEARNING_RATE])
        if BATCH_SIZE:
            server_command.extend(["-bs", BATCH_SIZE])
        if DIRICHLET_ALPHA:
            server_command.extend(["-diri-alpha", DIRICHLET_ALPHA])
        if PORT:
            server_command.extend(["-p", PORT])
        if SERVER_MODEL_AGGREGATION:
            server_command.extend(["-sma", SERVER_MODEL_AGGREGATION])
        if OPTIMIZER:
            server_command.extend(["-o", OPTIMIZER])
        if DELETE_FRACTION_OF_DATA:
            server_command.extend(["-df", DELETE_FRACTION_OF_DATA])
        if NETWORKING_FAIRNESS:
            if NETWORKING_FAIRNESS.lower() == "true":
                server_command.extend(["-nf"])
            elif NETWORKING_FAIRNESS.lower() == "false":
                server_command.extend(["-nnf"])

        client_commands = []
        if METHOD != "cl":
            client_commands = [
                [
                    "python3",
                    "client/main.py",
                    "-cid",
                    str(i),
                    "-d",
                    DEVICE,
                    "-su",
                    f"localhost:{PORT}",
                ]
                for i in range(int(TOTAL_CLIENTS))
            ]

        run_simulation(server_command, client_commands)
