from dataclasses import dataclass


@dataclass
class Paths:
    data: str
    project_name: str


@dataclass
class Params:
    epochs: int
    batch_size: int
    output_dir: str   
    learning_rate: float
    max_steps: int
    warmup_steps: int
    weight_decay: float
    logging_steps: int
    eval_steps: int
    load_best_model_at_end: bool
    tf32: bool
    metric_for_best_model: str
    # gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    save_total_limit: int


@dataclass
class Run:
    clean: str 
    num_classes: int                      
    raytune: str
    use_MLM: str
    hf_model: str
    struc: str
    exam: str
    ros: str
    rot: str
    ih: str
    ap: str
    sec: str
    


@dataclass
class TOXConfig:
    paths: Paths
    params: Params
    run: Run


# def yaml2config(path:str) -> dict:
#     with open(path, "r") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     return {i:config[i]['value'] for i in config}
