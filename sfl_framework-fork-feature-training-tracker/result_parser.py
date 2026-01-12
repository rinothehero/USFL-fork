import json
import os

import matplotlib.pyplot as plt

def parse_data(data):
    graph_result = {
        "accuracy": [],
        "networking": [],
        "time_acc": [],
        "time_top_1_acc": [],
        "top_1_acc": [],
    }
    result = {}

    metric = data["metric"]
    config = data["config"]
    method = config["method"]

    top_1_acc = 0
    networking = 0
    training_time = 0
    total_selected = 0
    total_submitted = 0

    time_to_top_1 = 0
    accuracy_time_map = {i: None for i in range(10, 101, 5)}

    for round_num in metric:
        if method == "prunefl" and round_num == "1":
            continue

        round_events = metric[round_num]

        round_start_time = 0
        round_end_time = 0

        round_selected_clients = 0
        round_submitted_clients = 0
        round_accuracy = 0

        round_networking = 0
        top_1_renew_flag = False

        prunefl_selected_clients = []
        prunefl_submitted_clients = []

        for event in round_events:
            event_name = event["event"]
            params = event.get("params", {})

            if method == "prunefl" and event_name == "SEND_MODEL_START":
                client_id = params.get("to")
                prunefl_selected_clients.append(client_id)
                round_start_time = event["timestamp"]

            if method == "prunefl" and event_name == "MODEL_RECIEVED":
                client_id = params.get("client_id")
                prunefl_submitted_clients.append(client_id)

            if event_name == "CLIENTS_SELECTED":
                round_start_time = event["timestamp"]
                client_ids = params.get("client_ids", [])
                round_selected_clients = len(client_ids)

            if event_name == "POST_ROUND_END":
                round_end_time = event["timestamp"]

            if event_name == "MODEL_EVALUATED":
                accuracy_val = params.get("accuracy", 0)
                if isinstance(accuracy_val, list):
                    accuracy = accuracy_val[0]["accuracy"]
                else:
                    accuracy = accuracy_val
                if accuracy > top_1_acc:
                    top_1_acc = accuracy
                    top_1_renew_flag = True
                round_accuracy = accuracy

            if event_name == "MODEL_AGGREGATION_START":
                client_ids = params.get("client_ids", [])
                round_submitted_clients = len(client_ids)

            if event_name in ["SEND_MODEL_START", "MODEL_RECEIVED"]:
                size = params.get("size", 0)
                round_networking += size

        if method == "prunefl":
            round_selected_clients = len(prunefl_selected_clients)
            round_submitted_clients = len(prunefl_submitted_clients)

        training_time += round_end_time - round_start_time
        networking += round_networking
        graph_result["accuracy"].append({"x": int(round_num), "y": round_accuracy})
        graph_result["networking"].append({"x": int(round_num), "y": round_networking})
        graph_result["time_acc"].append({"x": training_time, "y": round_accuracy})
        graph_result["time_top_1_acc"].append({"x": training_time, "y": top_1_acc})
        graph_result["top_1_acc"].append({"x": int(round_num), "y": top_1_acc})
        total_selected += round_selected_clients
        total_submitted += round_submitted_clients

        if top_1_renew_flag:
            top_1_renew_flag = False
            time_to_top_1 = training_time

        for threshold in accuracy_time_map:
            if (
                float(round_accuracy) * 100 >= threshold
                and accuracy_time_map[threshold] is None
            ):
                accuracy_time_map[threshold] = training_time

    result["top_1_acc"] = top_1_acc
    result["networking"] = networking
    result["training_time"] = training_time
    result["accuracy_time_map"] = accuracy_time_map
    result["time_to_top_1"] = time_to_top_1
    return result, graph_result


import json
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt


def load_all_json_files(folder_path):
    json_data = {}        
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    json_data[filename] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
    return json_data


def animate(
    i, folder_path, graph_key, x_label, y_label, title, graph_results, file_mod_times
):
    all_json_data = load_all_json_files(folder_path)
    updated = False

    for name, data in all_json_data.items():
        file_path = os.path.join(folder_path, name)
        mod_time = os.path.getmtime(file_path)

        # 파일이 새로 추가되었거나 수정된 경우 업데이트
        if name not in file_mod_times or file_mod_times[name] != mod_time:
            file_mod_times[name] = mod_time
            result, graph_result = parse_data(data)
            print(f"{name} 업데이트됨: {result}")
            graph_results[name] = graph_result
            updated = True

    if updated:
        plt.cla()  # 현재 그래프를 지웁니다.
        max_x_value = 0
        for label, data in graph_results.items():
            x_values = [point["x"] for point in data[graph_key]]
            y_values = [point["y"] for point in data[graph_key]]
            plt.plot(x_values, y_values, linestyle="-", label=label)
            if x_values and max(x_values) > max_x_value:
                max_x_value = max(x_values)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        if graph_key != "networking":
            plt.ylim(0, 1)
        plt.xlim(0, max_x_value)
        plt.tight_layout()


def main():
    folder_path = "."
    graph_results = {}
    file_mod_times = {}
    
    # 그릴 그래프의 키를 선택합니다.
    graph_key = "accuracy"  # "accuracy", "networking", "time_acc", "time_top_1_acc" 중에서 선택

    # 선택한 그래프 키에 따라 x축과 y축 레이블을 설정합니다.
    if graph_key == "accuracy":
        x_label = "Round"
        y_label = "Accuracy"
        title = "Round vs Accuracy"
    elif graph_key == "networking":
        x_label = "Round"
        y_label = "Networking"
        title = "Round vs Networking"
    elif graph_key == "time_acc":
        x_label = "Time (seconds)"
        y_label = "Accuracy"
        title = "Time vs Accuracy"
    elif graph_key == "time_top_1_acc":
        x_label = "Time (seconds)"
        y_label = "Top 1 Accuracy"
        title = "Time vs Top 1 Accuracy"
    else:
        x_label = "X"
        y_label = "Y"
        title = ""

    fig = plt.figure(figsize=(8, 6))
    animate(0, folder_path,
            graph_key,
            x_label,
            y_label,
            title,
            graph_results,
            file_mod_times,)
    global ani
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        fargs=(
            folder_path,
            graph_key,
            x_label,
            y_label,
            title,
            graph_results,
            file_mod_times,
        ),
        interval=5000,  # 5초마다 업데이트
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
