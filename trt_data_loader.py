import numpy as np

def load_data():
    data = np.load("bert-base-uncased/case_data.npz")

    input_ids = data["input_ids"].astype(np.int64)
    token_type_ids = data["token_type_ids"].astype(np.int64)
    position_ids = data["position_ids"].astype(np.int64)
    input_list = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }

    yield input_list