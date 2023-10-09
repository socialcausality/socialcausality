import argparse
import json
import os
from collections import namedtuple


def get_train_args():
    parser = argparse.ArgumentParser(description="AutoBots")
    # Section: General Configuration
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment identifier")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory for saving results")

    # Section: Dataset
    parser.add_argument("--train_data_size", type=int, default=-1, help="Number of scenes to train on.")
    parser.add_argument("--dataset", type=str, default="synth", choices=["Argoverse", "Nuscenes", "trajnet++",
                                                                       "interaction-dataset", "synth"],
                        help="Dataset to train on.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset files.")
    parser.add_argument("--use-map-image", type=bool, default=False, help="Use map image if applicable.")
    parser.add_argument("--use-map-lanes", type=bool, default=False, help="Use map lanes if applicable.")

    # Section: Algorithm
    parser.add_argument("--model-type", type=str, default="Autobot-Ego", choices=["Autobot-Joint", "Autobot-Ego"],
                        help="Whether to train for joint prediction or ego-only prediction.")
    parser.add_argument("--reg-type", type=str, default="None", choices=["None", "contrastive", "consistency", "ranking"],
                        help="Whether to train causality with consistency or contrastive regularizer.")
    parser.add_argument("--num-modes", type=int, default=1, help="Number of discrete latent variables for Autobot.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model's hidden size.")
    parser.add_argument("--num-encoder-layers", type=int, default=2,
                        help="Number of social-temporal layers in Autobot's encoder.")
    parser.add_argument("--num-decoder-layers", type=int, default=2,
                        help="Number of social-temporal layers in Autobot's decoder.")
    parser.add_argument("--tx-hidden-size", type=int, default=384,
                        help="hidden size of transformer layers' feedforward network.")
    parser.add_argument("--tx-num-heads", type=int, default=16, help="Transformer number of heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout strenght used throughout model.")

    # Section: Loss Function
    parser.add_argument("--entropy-weight", type=float, default=40.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--consistency-weight", type=float, default=1.0, metavar="lamda", help="Weight of consistency loss.")
    parser.add_argument("--contrastive-weight", type=float, default=1000.0, metavar="lamda", help="Weight of contrastive loss.")
    parser.add_argument("--ranking-weight", type=float, default=1000.0, metavar="lamda", help="Weight of ranking loss.")
    parser.add_argument("--ranking-margin", type=float, default=0.001, metavar="lamda", help="Margin of ranking loss.")
    parser.add_argument("--kl-weight", type=float, default=20.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--use-FDEADE-aux-loss", type=bool, default=True,
                        help="Whether to use FDE/ADE auxiliary loss in addition to NLL (accelerates learning).")

    # Section: Loading model
    parser.add_argument("--weight-path", type=str, default="", help="Path to model and optimizer weight path.")
    parser.add_argument("--start-epoch", type=int, default=0, help="Starting epoch from.")

    # Section: Training params:
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.00075, help="Learning rate")
    parser.add_argument("--adam-epsilon", type=float, default=1e-4, help="Adam optimiser epsilon value")
    parser.add_argument("--learning-rate-sched", type=int, nargs='+', default=[50, 100, 150, 200, 250],
                        help="Learning rate Schedule.")
    parser.add_argument("--grad-clip-norm", type=float, default=5, metavar="C", help="Gradient clipping norm")
    parser.add_argument("--num-epochs", type=int, default=750, metavar="I",
                        help="number of iterations through the dataset.")
    parser.add_argument("--save-every", type=int, default=50, help="Frequency of saving model.")
    parser.add_argument("--val-every", type=int, default=50, help="Frequency of validating model.")

    # Section: Evaluating
    parser.add_argument("--evaluate_causal", action="store_true", help="Evaluates causality understanding metrics.")
    args = parser.parse_args()

    if args.use_map_image and args.use_map_lanes:
        raise Exception('We do not support having both the map image and the map lanes...')

    # Perform config checks
    if "trajnet" in args.dataset:
        # assert "Joint" in args.model_type, "Can't run AutoBot-Ego on TrajNet..."
        assert not args.use_map_image and not args.use_map_lanes, "TrajNet++ has no scene map information..."
    elif "Argoverse" in args.dataset:
        assert "Ego" in args.model_type, "Can't run AutoBot-Joint on Argoverse..."
    elif "interaction-dataset" in args.dataset:
        assert "Ego" not in args.model_type, "Can't run AutoBot-Ego on Interaction Dataset..."
        assert not args.use_map_image, "Interaction-dataset does not have image-based scene information..."
    elif args.dataset == "synth":
        assert "Ego" in args.model_type, "Can't run AutoBot-Joint on Synth-v1..."
        assert not args.use_map_image and not args.use_map_lanes, "Synth-v1 has no scene map information..."

    results_dirname = create_results_folder(args)
    save_config(args, results_dirname)

    return args, results_dirname


def get_eval_args():
    parser = argparse.ArgumentParser(description="AutoBot")
    parser.add_argument("--models-path", type=str, required=True, help="Load model checkpoint")
    parser.add_argument("--dataset", type=str, default="synth", choices=["Argoverse", "Nuscenes", "trajnet++",
                                                                       "interaction-dataset", "synth"], help="Dataset to evaluate on.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path.")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--evaluate_causal", action="store_true", help="Evaluates causality understanding metrics.")
    args = parser.parse_args()

    config, model_dirname = load_config(args.models_path)
    config = namedtuple("config", config.keys())(*config.values())
    return args, config, model_dirname


def create_results_folder(args):
    model_configname = ""
    model_configname += "Autobot_joint" if "Joint" in args.model_type else "Autobot_ego"
    # model_configname += "_C"+str(args.num_modes) + "_H"+str(args.hidden_size) + "_E"+str(args.num_encoder_layers)
    # model_configname += "_D"+str(args.num_decoder_layers) + "_TXH"+str(args.tx_hidden_size) + "_NH"+str(args.tx_num_heads)
    # model_configname += "_EW"+str(int(args.entropy_weight)) + "_KLW"+str(int(args.kl_weight))
    # model_configname += "_NormLoss" if args.use_FDEADE_aux_loss else ""
    # model_configname += "_roadImg" if args.use_map_image else ""
    # model_configname += "_roadLanes" if args.use_map_lanes else ""
    # model_configname += "_NScene:" + str(args.train_data_size)
    model_configname += "_regType:" + args.reg_type
    if args.reg_type == "consistency":
        model_configname += "_W:" + str(int(args.consistency_weight))
    elif args.reg_type == "contrastive":
        model_configname += "_W:" + str(int(args.contrastive_weight))
    elif args.reg_type == "ranking":
        model_configname += "_W:" + str(int(args.ranking_weight)) + "_M:" + str(args.ranking_margin)
    if args.exp_id is not None:
        model_configname += ("_" + args.exp_id)
    model_configname += "_s"+str(args.seed)

    result_dirname = os.path.join(args.save_dir, "results", args.dataset, model_configname)
    if os.path.isdir(result_dirname):
        answer = input(result_dirname + " exists. \n Do you wish to overwrite? (y/n)")
        if 'y' in answer:
            if os.path.isdir(os.path.join(result_dirname, "tb_files")):
                for f in os.listdir(os.path.join(result_dirname, "tb_files")):
                    os.remove(os.path.join(result_dirname, "tb_files", f))
        else:
            exit()
    os.makedirs(result_dirname, exist_ok=True)
    return result_dirname


def save_config(args, results_dirname):
    argparse_dict = vars(args)
    with open(os.path.join(results_dirname, 'config.json'), 'w') as fp:
        json.dump(argparse_dict, fp)


def load_config(model_path):
    model_dirname = os.path.join(*model_path.split("/")[:-1])
    if model_path[0] == "/":
        model_dirname = "/" + model_dirname
    assert os.path.isdir(model_dirname)
    with open(os.path.join(model_dirname, 'config.json'), 'r') as fp:
        config = json.load(fp)
    return config, model_dirname
