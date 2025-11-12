import random

import torch


def combine_data(data, num_frames=57, keyboard_dim=6, mouse=True):
    assert num_frames % 4 == 1
    keyboard_condition = torch.zeros((num_frames, keyboard_dim))
    if mouse:
        mouse_condition = torch.zeros((num_frames, 2))

    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]["keyboard_condition"]
        if mouse:
            m = data[rd]["mouse_condition"]

        if current_frame:
            keyboard_condition[:1] = k[:1]
            if mouse:
                mouse_condition[:1] = m[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            keyboard_condition[current_frame : current_frame + rd_frame] = k.repeat(repeat_time, 1)
            if mouse:
                mouse_condition[current_frame : current_frame + rd_frame] = m.repeat(repeat_time, 1)
            current_frame += rd_frame
    if mouse:
        return {"keyboard_condition": keyboard_condition, "mouse_condition": mouse_condition}
    return {"keyboard_condition": keyboard_condition}


def Bench_actions_universal(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        # "back",
        "left",
        "right",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
        # "back_left",
        # "back_right",
    ]

    actions_single_camera = [
        "camera_l",
        "camera_r",
        # "camera_ur",
        # "camera_ul",
        # "camera_dl",
        # "camera_dr"
        # "camera_up",
        # "camera_down",
    ]
    actions_to_test = actions_double_action * 5 + actions_single_camera * 5 + actions_single_action * 5
    for action in actions_single_action + actions_double_action:
        for camera in actions_single_camera:
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = {"forward": 0, "back": 1, "left": 2, "right": 3}

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_up": [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l": [0, -CAM_VALUE],
        "camera_r": [0, CAM_VALUE],
        "camera_ur": [CAM_VALUE, CAM_VALUE],
        "camera_ul": [CAM_VALUE, -CAM_VALUE],
        "camera_dr": [-CAM_VALUE, CAM_VALUE],
        "camera_dl": [-CAM_VALUE, -CAM_VALUE],
    }

    data = []

    for action_name in actions_to_test:
        keyboard_condition = [[0, 0, 0, 0] for _ in range(num_samples_per_action)]
        mouse_condition = [[0, 0] for _ in range(num_samples_per_action)]

        for sub_act in base_action:
            if sub_act not in action_name:  # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act] for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({"keyboard_condition": torch.tensor(keyboard_condition), "mouse_condition": torch.tensor(mouse_condition)})
    return combine_data(data, num_frames, keyboard_dim=4, mouse=True)


def Bench_actions_gta_drive(num_frames, num_samples_per_action=4):
    actions_single_action = [
        "forward",
        "back",
    ]

    actions_single_camera = [
        "camera_l",
        "camera_r",
    ]
    actions_to_test = actions_single_camera * 2 + actions_single_action * 2
    for action in actions_single_action:
        for camera in actions_single_camera:
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    # print("length of actions: ", len(actions_to_test))
    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = {"forward": 0, "back": 1}

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_l": [0, -CAM_VALUE],
        "camera_r": [0, CAM_VALUE],
    }

    data = []

    for action_name in actions_to_test:
        keyboard_condition = [[0, 0] for _ in range(num_samples_per_action)]
        mouse_condition = [[0, 0] for _ in range(num_samples_per_action)]

        for sub_act in base_action:
            if sub_act not in action_name:  # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [CAMERA_VALUE_MAP[sub_act] for _ in range(num_samples_per_action)]

            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({"keyboard_condition": torch.tensor(keyboard_condition), "mouse_condition": torch.tensor(mouse_condition)})
    return combine_data(data, num_frames, keyboard_dim=2, mouse=True)


def Bench_actions_templerun(num_frames, num_samples_per_action=4):
    actions_single_action = ["jump", "slide", "leftside", "rightside", "turnleft", "turnright", "nomove"]

    actions_to_test = actions_single_action

    base_action = actions_single_action

    KEYBOARD_IDX = {"nomove": 0, "jump": 1, "slide": 2, "turnleft": 3, "turnright": 4, "leftside": 5, "rightside": 6}

    data = []

    for action_name in actions_to_test:
        keyboard_condition = [[0, 0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)]

        for sub_act in base_action:
            if sub_act not in action_name:  # 只处理action_name包含的动作
                continue
            # print(f"action name: {action_name} sub_act: {sub_act}")
            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({"keyboard_condition": torch.tensor(keyboard_condition)})
    return combine_data(data, num_frames, keyboard_dim=7, mouse=False)


class MatrixGame2_Bench:
    def __init__(self):
        self.deivce = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

    def get_conditions(self, mode, num_frames):
        conditional_dict = {}
        if mode == "universal":
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict["mouse_cond"] = mouse_condition
        elif mode == "gta_drive":
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict["mouse_cond"] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data["keyboard_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict["keyboard_cond"] = keyboard_condition
        return conditional_dict
