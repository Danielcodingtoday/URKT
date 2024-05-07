import os
import json
from argparse import ArgumentParser

import torch
import tomlkit
from tqdm import tqdm


import sys
sys.path.append("/home/q22301155/codedemo/KT/Uncertain_V1")
DATA_DIR = "/home/q22301155/codedemo/KT/DTransformer-main_idea_copy/data"
from DTransformer.data import KTData
from DTransformer.eval import Evaluator
# configure the main parser
parser = ArgumentParser()
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=64, type=int)

# data setup
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)
parser.add_argument(
    "-p", "--with_pid", help="provide model with pid", action="store_true"
)

# model setup
# TODO: model size, dropout rate, etc.
parser.add_argument("-m", "--model", help="choose model")
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=1)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument(
    "--n_know", help="dimension of knowledge parameter", type=int, default=32
)

# test setup
parser.add_argument("-f_c", "--from_file_c", help="test existing model file for certain", required=True)
parser.add_argument("-f_uc", "--from_file_uc", help="test existing model file for uncertain", required=True)
parser.add_argument("-N", help="T+N prediction window size", type=int, default=1)


# testing logic
def main(args):
    # prepare datasets
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    test_data = KTData(
        os.path.join(DATA_DIR, dataset["test"]),
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
    )

    # prepare model
    if args.model == "DKT":
        from baselines.DKT import DKT

        model = DKT(dataset["n_questions"], args.d_model)
    elif args.model == "DKVMN":
        from baselines.DKVMN import DKVMN

        model = DKVMN(dataset["n_questions"], args.batch_size)
    elif args.model == "AKT":
        from baselines.AKT import AKT

        model = AKT(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_heads=args.n_heads,
        )
    else:
        from DTransformer.modelset import DTransformer

        model_c = DTransformer(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_know=args.n_know,
            n_layers=args.n_layers,
            sample_num=10
        )
        # model_uc = DTransformer(
        #     dataset["n_questions"],
        #     dataset["n_pid"],
        #     d_model=args.d_model,
        #     n_heads=args.n_heads,
        #     n_know=args.n_know,
        #     n_layers=args.n_layers,
        #     sample_num=30
        # )

    model_c.load_state_dict(torch.load(args.from_file_c, map_location=lambda s, _: s))
    model_c.to(args.device)
    model_c.eval()
    ##不确定性模型
    # model_uc.load_state_dict(torch.load(args.from_file_uc, map_location=lambda s, _: s))
    # model_uc.to(args.device)
    # model_uc.eval()

    # test
    evaluator = Evaluator()


    with torch.no_grad():
        it = tqdm(iter(test_data))
        for batch in it:
            if args.with_pid:
                q, s, pid = batch.get("q", "s", "pid")
            else:
                q, s = batch.get("q", "s")
                pid = None if seq_len is None else [None] * len(q)
            if seq_len is None:
                q, s, pid = [q], [s], [pid]
            for q, s, pid in zip(q, s, pid):
                q = q.to(args.device)
                s = s.to(args.device)
                if pid is not None:
                    pid = pid.to(args.device)
                y_c,logits_mean_c,_,y_kc_c,_ = model_c.predict(q, s, pid, args.N)
                y_c,logits_mean_uc,_,y_kc_c,_ = model_c.predict(q, s, pid, args.N)
                
                # y_uc,logits_mean_uc, _,y_kc_uc,_  = model_uc.predict(q, s, pid, args.N)
                
                evaluator.evaluate(s[:, (args.N - 1) :], (torch.sigmoid(logits_mean_c)+torch.sigmoid(logits_mean_uc))/2)




            # it.set_postfix(evaluator.report())

    output_path = args.from_file_c + ".json"
    if os.path.exists(output_path):
        output = json.load(open(output_path))
    else:
        output = {"args": vars(args), "metrics": {}}

    output["metrics"][args.N] = evaluator.report()
    print(output["metrics"][args.N])

    json.dump(output, open(output_path, "w"), indent=2)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
