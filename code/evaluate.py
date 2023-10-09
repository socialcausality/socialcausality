import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from datasets.argoverse.dataset import ArgoH5Dataset
from datasets.interaction_dataset.dataset import InteractionDataset
from datasets.nuscenes.dataset import NuscenesH5Dataset
from datasets.synth.dataset import SynthV1CausalDataset, my_collate_fn
from datasets.trajnetpp.dataset import TrajNetPPDataset
from models.autobot_ego import AutoBotEgo
from models.autobot_joint import AutoBotJoint
from process_args import get_eval_args
from utils.metric_helpers import min_xde_K, yaw_from_predictions, interpolate_trajectories, collisions_for_inter_dataset, collision_rate
from utils.train_helpers import  calc_consistency_loss, HNC_ARS


class Evaluator:
    def __init__(self, args, model_config, model_dirname):
        self.args = args
        self.model_config = model_config
        self.model_dirname = model_dirname
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        torch.manual_seed(self.model_config.seed)
        if torch.cuda.is_available() and not self.args.disable_cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.model_config.seed)
        else:
            self.device = torch.device("cpu")

        self.interact_eval = False  # for evaluating on the interaction dataset, we need this bool.
        self.initialize_dataloader()
        self.initialize_model()

    def initialize_dataloader(self):
        if "Nuscenes" in self.args.dataset:
            val_dset = NuscenesH5Dataset(dset_path=self.args.dataset_path, split_name="val",
                                         model_type=self.model_config.model_type,
                                         use_map_img=self.model_config.use_map_image,
                                         use_map_lanes=self.model_config.use_map_lanes)

        elif "interaction-dataset" in self.args.dataset:
            val_dset = InteractionDataset(dset_path=self.args.dataset_path, split_name="val",
                                          use_map_lanes=self.model_config.use_map_lanes, evaluation=True)
            self.interact_eval = True

        elif "trajnet++" in self.args.dataset:
            val_dset = TrajNetPPDataset(dset_path=self.args.dataset_path, split_name="test")

        elif "Argoverse" in self.args.dataset:
            val_dset = ArgoH5Dataset(dset_path=self.args.dataset_path, split_name="val",
                                     use_map_lanes=self.model_config.use_map_lanes)

        elif self.args.dataset == "synth":
            val_dset = SynthV1CausalDataset(dset_path=self.args.dataset_path, split="val")

        else:
            raise NotImplementedError

        self.num_other_agents = val_dset.num_others
        self.pred_horizon = val_dset.pred_horizon
        self.k_attr = val_dset.k_attr
        self.map_attr = val_dset.map_attr
        self.predict_yaw = val_dset.predict_yaw
        if "Joint" in self.model_config.model_type:
            self.num_agent_types = val_dset.num_agent_types

        if self.args.dataset == "synth":
            self.val_loader = torch.utils.data.DataLoader(
                val_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False,
                pin_memory=False, collate_fn=my_collate_fn
            )
        else:
            self.val_loader = torch.utils.data.DataLoader(
                val_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False,
                pin_memory=False
            )

        print("Val dataset loaded with length", len(val_dset))

    def initialize_model(self):
        if "Ego" in self.model_config.model_type:
            self.autobot_model = AutoBotEgo(k_attr=self.k_attr,
                                            d_k=self.model_config.hidden_size,
                                            _M=self.num_other_agents,
                                            c=self.model_config.num_modes,
                                            T=self.pred_horizon,
                                            L_enc=self.model_config.num_encoder_layers,
                                            dropout=self.model_config.dropout,
                                            num_heads=self.model_config.tx_num_heads,
                                            L_dec=self.model_config.num_decoder_layers,
                                            tx_hidden_size=self.model_config.tx_hidden_size,
                                            use_map_img=self.model_config.use_map_image,
                                            use_map_lanes=self.model_config.use_map_lanes,
                                            map_attr=self.map_attr,
                                            return_embeddings=(self.model_config.reg_type in ["contrastive", "ranking"] and self.model_config.dataset == "synth")).to(self.device)

        elif "Joint" in self.model_config.model_type:
            self.autobot_model = AutoBotJoint(k_attr=self.k_attr,
                                              d_k=self.model_config.hidden_size,
                                              _M=self.num_other_agents,
                                              c=self.model_config.num_modes,
                                              T=self.pred_horizon,
                                              L_enc=self.model_config.num_encoder_layers,
                                              dropout=self.model_config.dropout,
                                              num_heads=self.model_config.tx_num_heads,
                                              L_dec=self.model_config.num_decoder_layers,
                                              tx_hidden_size=self.model_config.tx_hidden_size,
                                              use_map_lanes=self.model_config.use_map_lanes,
                                              map_attr=self.map_attr,
                                              num_agent_types=self.num_agent_types,
                                              predict_yaw=self.predict_yaw).to(self.device)
        else:
            raise NotImplementedError

        model_dicts = torch.load(self.args.models_path, map_location=self.device)
        self.autobot_model.load_state_dict(model_dicts["AutoBot"])
        self.autobot_model.eval()

        model_parameters = filter(lambda p: p.requires_grad, self.autobot_model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of Model Parameters:", num_params)

    def _data_to_device(self, data, model_type_overwrite=None):
        model_type = self.model_config.model_type
        if model_type_overwrite is not None:
            model_type = model_type_overwrite

        if "Joint" in model_type:
            ego_in, ego_out, agents_in, agents_out, context_img, agent_types = data
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            agents_in = agents_in.float().to(self.device)
            agents_out = agents_out.float().to(self.device)
            context_img = context_img.float().to(self.device)
            agent_types = agent_types.float().to(self.device)
            return ego_in, ego_out, agents_in, agents_out, context_img, agent_types

        elif "Ego" in model_type:
            ego_in, ego_out, agents_in, roads = data
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            agents_in = agents_in.float().to(self.device)
            roads = roads.float().to(self.device)
            return ego_in, ego_out, agents_in, roads

    def _compute_ego_errors(self, ego_preds, ego_gt):
        ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
        ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1).transpose(0, 1).cpu().numpy()
        fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1).cpu().numpy()
        return ade_losses, fde_losses

    def _compute_marginal_errors(self, preds, ego_gt, agents_gt, agents_in):
        agent_masks = torch.cat((torch.ones((len(agents_in), 1)).to(self.device), agents_in[:, -1, :, -1]), dim=-1).view(1, 1, len(agents_in), -1)
        agent_masks[agent_masks == 0] = float('nan')
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2).unsqueeze(0).permute(0, 2, 1, 3, 4)
        error = torch.norm(preds[:, :, :, :, :2] - agents_gt[:, :, :, :, :2], 2, dim=-1) * agent_masks
        ade_losses = np.nanmean(error.cpu().numpy(), axis=1).transpose(1, 2, 0)
        fde_losses = error[:, -1].cpu().numpy().transpose(1, 2, 0)
        return ade_losses, fde_losses

    def _compute_joint_errors(self, preds, ego_gt, agents_gt):
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2)
        agents_masks = agents_gt[:, :, :, -1]
        agents_masks[agents_masks == 0] = float('nan')

        ade_losses = []
        for k in range(self.model_config.num_modes):
            ade_error = (torch.norm(preds[k, :, :, :, :2].transpose(0, 1) - agents_gt[:, :, :, :2], 2, dim=-1)
                         * agents_masks).cpu().numpy()
            ade_error = np.nanmean(ade_error, axis=(1, 2))
            ade_losses.append(ade_error)
        ade_losses = np.array(ade_losses).transpose()

        fde_losses, ade_losses = []
        for k in range(self.model_config.num_modes):
            fde_error = (torch.norm(preds[k, -1, :, :, :2] - agents_gt[:, -1, :, :2], 2, dim=-1) * agents_masks[:, -1]).cpu().numpy()
            fde_error = np.nanmean(fde_error, axis=1)
            fde_losses.append(fde_error)
        fde_losses = np.array(fde_losses).transpose()

        return ade_losses, fde_losses

    def autobotjoint_evaluate(self):
        with torch.no_grad():
            val_marg_ade_losses = []
            val_marg_fde_losses = []
            val_marg_mode_probs = []
            val_scene_ade_losses = []
            val_scene_fde_losses = []
            val_mode_probs = []
            if self.interact_eval:
                total_collisions = []
            for i, data in enumerate(self.val_loader):
                if i % 50 == 0:
                    print(i, "/", len(self.val_loader.dataset) // self.args.batch_size)

                if self.interact_eval:
                    # for the interaction dataset, we have multiple outputs that we use to interpolate, rotate and
                    # compute scene collisions almost like they do.
                    orig_ego_in, orig_agents_in, original_roads, translations = data[6:]
                    data = data[:6]
                    orig_ego_in = orig_ego_in.float().to(self.device)
                    orig_agents_in = orig_agents_in.float().to(self.device)

                ego_in, ego_out, agents_in, agents_out, context_img, agent_types = self._data_to_device(data)
                pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, context_img, agent_types)

                if self.interact_eval:
                    pred_obs = interpolate_trajectories(pred_obs)
                    pred_obs = yaw_from_predictions(pred_obs, orig_ego_in, orig_agents_in)
                    scene_collisions, pred_obs, vehicles_only = collisions_for_inter_dataset(pred_obs.cpu().numpy(),
                                                                                             agent_types.cpu().numpy(),
                                                                                             orig_ego_in.cpu().numpy(),
                                                                                             orig_agents_in.cpu().numpy(),
                                                                                             translations.cpu().numpy(),
                                                                                             device=self.device)
                    total_collisions.append(scene_collisions)

                # Marginal metrics
                ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in)
                val_marg_ade_losses.append(ade_losses.reshape(-1, self.model_config.num_modes))
                val_marg_fde_losses.append(fde_losses.reshape(-1, self.model_config.num_modes))
                val_marg_mode_probs.append(
                    mode_probs.unsqueeze(1).repeat(1, self.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                        -1, self.model_config.num_modes))

                # Joint metrics
                scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out)
                val_scene_ade_losses.append(scene_ade_losses)
                val_scene_fde_losses.append(scene_fde_losses)
                val_mode_probs.append(mode_probs.detach().cpu().numpy())

            val_marg_ade_losses = np.concatenate(val_marg_ade_losses)
            val_marg_fde_losses = np.concatenate(val_marg_fde_losses)
            val_marg_mode_probs = np.concatenate(val_marg_mode_probs)

            val_scene_ade_losses = np.concatenate(val_scene_ade_losses)
            val_scene_fde_losses = np.concatenate(val_scene_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)

            val_minade_c = min_xde_K(val_marg_ade_losses, val_marg_mode_probs, K=self.model_config.num_modes)
            val_minfde_c = min_xde_K(val_marg_fde_losses, val_marg_mode_probs, K=self.model_config.num_modes)
            val_sminade_c = min_xde_K(val_scene_ade_losses, val_mode_probs, K=self.model_config.num_modes)
            val_sminfde_c = min_xde_K(val_scene_fde_losses, val_mode_probs, K=self.model_config.num_modes)

            print("Marg. minADE c:", val_minade_c[0], "Marg. minFDE c:", val_minfde_c[0])
            print("Scene minADE c:", val_sminade_c[0], "Scene minFDE c:", val_sminfde_c[0])

            if self.interact_eval:
                total_collisions = np.concatenate(total_collisions).mean()
                print("Scene Collision Rate", total_collisions)

    def autobotego_evaluate(self):
        with torch.no_grad():
            val_ade_losses = []
            val_fde_losses = []
            val_mode_probs = []
            val_collision_rates = []
            if self.args.evaluate_causal:
                val_consistency = []
                val_HNC, val_ARS = 0, []
            for i, data in enumerate(self.val_loader):
                if i % 50 == 0:
                    print(i, "/", len(self.val_loader.dataset) // self.args.batch_size)

                if self.args.dataset == "synth":
                    scenes, causal_effects, data_splits = data
                    if not self.args.evaluate_causal:
                        scenes = [data[data_splits[:-1]] for data in scenes]
                    ego_in, ego_out, agents_in, agents_out, context_img, _ = self._data_to_device(scenes, "Joint")
                    roads = context_img
                    causal_effects = [torch.Tensor(causal_effect).float().to(self.device) for causal_effect in causal_effects]
                elif self.args.dataset == "trajnet++":
                    ego_in, ego_out, agents_in, agents_out, context_img, _ = self._data_to_device(data, "Joint")
                    roads = context_img
                else:
                    ego_in, ego_out, agents_in, roads = self._data_to_device(data)

                if "Ego" in self.model_config.model_type:
                    if self.model_config.dataset == "synth" and self.model_config.reg_type == "contrastive":
                        pred_obs, mode_probs, _ = self.autobot_model(ego_in, agents_in, roads)
                    else:
                        pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, roads)
                elif "Joint" in self.model_config.model_type:
                    pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, context_img, agent_types)
                    pred_obs = pred_obs[:, :, :, 0, :]
                else:
                    raise ValueError
                if self.args.evaluate_causal:
                    ade_losses, fde_losses = self._compute_ego_errors(pred_obs[:, :, data_splits[:-1], :], ego_out[data_splits[:-1], :, :2])
                    consistency_loss = calc_consistency_loss(pred_obs, causal_effects, data_splits, 1)
                    batch_HNC, batch_ARS = HNC_ARS(pred_obs, causal_effects, data_splits)
                else:
                    ade_losses, fde_losses = self._compute_ego_errors(pred_obs, ego_out)
                if self.args.dataset in ["synth", "trajnet++"]:
                    coll_rate = collision_rate(pred_obs, agents_out)
                    val_collision_rates.append(coll_rate)

                val_ade_losses.append(ade_losses)
                val_fde_losses.append(fde_losses)
                if self.args.evaluate_causal:
                    val_mode_probs.append(mode_probs[data_splits[:-1]].detach().cpu().numpy())
                    val_consistency.append(consistency_loss.item())
                    val_HNC += batch_HNC
                    val_ARS += batch_ARS
                else:
                    val_mode_probs.append(mode_probs.detach().cpu().numpy())

            val_ade_losses = np.concatenate(val_ade_losses)
            val_fde_losses = np.concatenate(val_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)
            if self.args.evaluate_causal:
                val_ARS = np.concatenate(val_ARS).mean()

            val_minade_c = min_xde_K(val_ade_losses, val_mode_probs, K=self.model_config.num_modes)
            val_minade_10 = min_xde_K(val_ade_losses, val_mode_probs, K=min(self.model_config.num_modes, 10))
            val_minade_5 = min_xde_K(val_ade_losses, val_mode_probs, K=5)
            val_minfde_c = min_xde_K(val_fde_losses, val_mode_probs, K=self.model_config.num_modes)
            val_minfde_1 = min_xde_K(val_fde_losses, val_mode_probs, K=1)

            if self.args.evaluate_causal:
                print("minADE_{}:".format(self.model_config.num_modes), val_minade_c[0],
                      "minADE_10", val_minade_10[0], "minADE_5", val_minade_5[0],
                      "minFDE_{}:".format(self.model_config.num_modes), val_minfde_c[0], "minFDE_1:", val_minfde_1[0],
                      "Consistency:", round(np.array(val_consistency).mean(), 2), "HNC:", val_HNC, "ARS:", round(val_ARS, 2))
            else:
                print("minADE_{}:".format(self.model_config.num_modes), val_minade_c[0],
                      "minADE_10", val_minade_10[0], "minADE_5", val_minade_5[0],
                      "minFDE_{}:".format(self.model_config.num_modes), val_minfde_c[0], "minFDE_1:", val_minfde_1[0])
            if self.args.dataset in ["synth", "trajnet++"]:
                val_collision_rates = np.concatenate(val_collision_rates)
                print("Collision Rate:", val_collision_rates.mean())


    def evaluate(self):
        if "Joint" in self.model_config.model_type:
            print(" EGO")
            self.autobotego_evaluate()
            print(" JOINT")
            self.autobotjoint_evaluate()
        elif "Ego" in self.model_config.model_type:
            print(" EGO")
            self.autobotego_evaluate()
        else:
            raise NotImplementedError


if __name__ == '__main__':
    args, config, model_dirname = get_eval_args()
    evaluator = Evaluator(args, config, model_dirname)
    evaluator.evaluate()
