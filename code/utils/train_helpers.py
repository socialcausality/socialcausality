import torch
from scipy import special
import numpy as np
import torch.distributions as D
from torch.distributions import MultivariateNormal, Laplace


# ==================================== AUTOBOT-EGO STUFF ====================================

def get_BVG_distributions(pred):
    B = pred.size(0)
    T = pred.size(1)
    mu_x = pred[:, :, 0].unsqueeze(2)
    mu_y = pred[:, :, 1].unsqueeze(2)
    sigma_x = pred[:, :, 2]
    sigma_y = pred[:, :, 3]
    rho = pred[:, :, 4]

    cov = torch.zeros((B, T, 2, 2)).to(pred.device)
    cov[:, :, 0, 0] = sigma_x ** 2
    cov[:, :, 1, 1] = sigma_y ** 2
    cov[:, :, 0, 1] = rho * sigma_x * sigma_y
    cov[:, :, 1, 0] = rho * sigma_x * sigma_y

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist


def get_Laplace_dist(pred):
    return Laplace(pred[:, :, :2], pred[:, :, 2:4])


def nll_pytorch_dist(pred, data, rtn_loss=True):
    # biv_gauss_dist = get_BVG_distributions(pred)
    biv_gauss_dist = get_Laplace_dist(pred)
    if rtn_loss:
        # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
        return (-biv_gauss_dist.log_prob(data)).sum(-1).sum(1)  # Laplace
    else:
        # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
        return (-biv_gauss_dist.log_prob(data)).sum(dim=(1, 2))  # Laplace


def nll_loss_multimodes(pred, data, modes_pred, entropy_weight=1.0, kl_weight=1.0, use_FDEADE_aux_loss=True):
    """NLL loss multimodes for training. MFP Loss function
    Args:
      pred: [K, T, B, 5]
      data: [B, T, 5]
      modes_pred: [B, K], prior prob over modes
      noise is optional
    """
    modes = len(pred)
    nSteps, batch_sz, dim = pred[0].shape

    # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
    log_lik = np.zeros((batch_sz, modes))
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=False)
            log_lik[:, kk] = -nll.cpu().numpy()

    priors = modes_pred.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors)
    log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
    post_pr = np.exp(log_posterior)
    post_pr = torch.tensor(post_pr).float().to(data.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()

    # Compute loss.
    loss = 0.0
    for kk in range(modes):
        nll_k = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=True) * post_pr[:, kk]
        loss += nll_k.mean()

    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions(pred[kk]).entropy())
    entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
    entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
    loss += entropy_weight * entropy_loss

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
    kl_loss = kl_weight*kl_loss_fn(torch.log(modes_pred), post_pr)

    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    if use_FDEADE_aux_loss:
        adefde_loss = l2_loss_fde(pred, data)
    else:
        adefde_loss = torch.tensor(0.0).to(data.device)

    return loss, kl_loss, post_entropy, adefde_loss


def l2_loss_fde(pred, data):
    fde_loss = torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1)
    ade_loss = torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2, dim=-1).mean(dim=2).transpose(0, 1)
    loss, min_inds = (fde_loss + ade_loss).min(dim=1)
    return 100.0 * loss.mean()


def calc_consistency_loss(pred_obs, causal_effects, data_splits, consistency_weight=1.0):
    consistency_diffs = []
    mse = torch.nn.MSELoss()
    for sample_id in range(len(causal_effects)):
        f_pred = pred_obs[0, :, data_splits[sample_id], :2]
        cf_preds = pred_obs[0, :, data_splits[sample_id] + 1:data_splits[sample_id + 1], :2]
        sensitivities = torch.norm(cf_preds - f_pred.unsqueeze(1), 2, dim=-1).mean(dim=0)

        consistency_diffs.append(mse(sensitivities, causal_effects[sample_id]))
    if len(consistency_diffs) == 0:
        return torch.Tensor(1)
    consistency_loss = torch.mean(torch.stack(consistency_diffs)) * consistency_weight
    return consistency_loss


def calc_contrastive_loss(embeds, causal_effects, data_splits, contrastive_weight=1.0, tau=0.2):
    infonces = []
    for sample_id in range(len(causal_effects)):
        q = embeds[data_splits[sample_id]]
        positives = np.where(causal_effects[sample_id].cpu().numpy() <= 0.02)[0]
        if len(positives) > 0:
            positive_id = np.random.choice(positives, 1)[0]
            k_plus = embeds[data_splits[sample_id] + 1 + positive_id]
            k_negs = embeds[data_splits[sample_id] + 1 + np.where(causal_effects[sample_id].cpu().numpy() >= 0.1)[0]]

            numerator = torch.matmul(q, k_plus) / tau
            denominator = torch.exp(numerator) + torch.sum(torch.exp(torch.matmul(k_negs, q) / tau))
            infonces.append(-numerator + torch.log(denominator))
    if len(infonces) == 0:
        return torch.Tensor(1)
    contrastive_loss = torch.mean(torch.stack(infonces)) * contrastive_weight
    # breakpoint()
    return contrastive_loss

def calc_ranking_loss(embeds, causal_effects, data_splits, ranking_weight=1.0, margin=0.001):
    ranking_losses = []
    for sample_id in range(len(causal_effects)):
        # Compute the distance of counterfactuals to the factual scene
        q = embeds[data_splits[sample_id]]
        keys = embeds[data_splits[sample_id] + 1:data_splits[sample_id + 1]]
        keys = keys[torch.argsort(causal_effects[sample_id])]
        dists = torch.matmul(keys, q)

        # Compute the difference of causal effect matrix
        ce = causal_effects[sample_id][torch.argsort(causal_effects[sample_id])]
        ce_row = ce.unsqueeze(1)
        ce_col = ce.unsqueeze(0)
        diff_ce = ce_col - ce_row

        # Create a mask for values between 0.2 and 0.5
        mask = (diff_ce >= 0.2) & (diff_ce <= 0.5)

        # Take the indices of the columns that are valid according to the mask
        cols = torch.arange(mask.size(1)).expand_as(mask).to(mask.device)
        valid_cols = cols[mask]

        # Calculate the number of valid columns per row which is the number of elements in the valid_cols for each row
        lengths = mask.sum(dim=1)
        if lengths.max().item() == 0:
            continue  # The mask is empty, skip this sample

        # Make the offsets for valid_cols, with offset[i] pointing out to the first element in valid_cols for row i
        offsets = torch.cat((torch.tensor([0]).to(mask.device), lengths.cumsum(dim=0)[:-1]))

        # Create a random index for each row and select the corresponding column from valid_cols
        rand_indices = torch.randint_like(lengths, 0, lengths.max().item()).to(mask.device)
        rand_indices = rand_indices % lengths
        selection = (offsets + rand_indices).long()
        selected_cols = valid_cols[selection[lengths > 0]]

        # Calculate the ranking loss
        ranking_losses.append(torch.nn.MarginRankingLoss(margin=margin)(dists[selected_cols], dists[lengths > 0], torch.ones((lengths > 0).sum()).to(dists.device)))

    if len(ranking_losses) == 0:
        return torch.Tensor(1)
    ranking_loss = torch.mean(torch.stack(ranking_losses)) * ranking_weight
    # breakpoint()
    return ranking_loss

def HNC_ARS(pred_obs, causal_effects, data_splits, non_causal_thresh=0.02, causal_thresh=0.1):
    HNC, ARS = 0, []
    for sample_id in range(len(causal_effects)):
        f_pred = pred_obs[0, :, data_splits[sample_id], :2]
        cf_preds = pred_obs[0, :, data_splits[sample_id] + 1:data_splits[sample_id + 1], :2]
        sensitivities = torch.norm(cf_preds - f_pred.unsqueeze(1), 2, dim=-1).mean(dim=0).detach().cpu().numpy()

        c_e = np.array(causal_effects[sample_id].cpu())

        HNC += (sensitivities[c_e < non_causal_thresh] > causal_thresh).sum()
        ARS.append(sensitivities[c_e < non_causal_thresh])
    return HNC, ARS

# ==================================== AUTOBOT-JOINT STUFF ====================================


def get_BVG_distributions_joint(pred):
    B = pred.size(0)
    T = pred.size(1)
    N = pred.size(2)
    mu_x = pred[:, :, :, 0].unsqueeze(3)
    mu_y = pred[:, :, :, 1].unsqueeze(3)
    sigma_x = pred[:, :, :, 2]
    sigma_y = pred[:, :, :, 3]
    rho = pred[:, :, :, 4]

    cov = torch.zeros((B, T, N, 2, 2)).to(pred.device)
    cov[:, :, :, 0, 0] = sigma_x ** 2
    cov[:, :, :, 1, 1] = sigma_y ** 2
    cov_val = rho * sigma_x * sigma_y
    cov[:, :, :, 0, 1] = cov_val
    cov[:, :, :, 1, 0] = cov_val

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist


def get_Laplace_dist_joint(pred):
    return Laplace(pred[:, :, :, :2], pred[:, :, :, 2:4])


def nll_pytorch_dist_joint(pred, data, agents_masks):
    # biv_gauss_dist = get_BVG_distributions_joint(pred)
    biv_gauss_dist = get_Laplace_dist_joint(pred)
    num_active_agents_per_timestep = agents_masks.sum(2)
    loss = (((-biv_gauss_dist.log_prob(data).sum(-1) * agents_masks).sum(2)) / num_active_agents_per_timestep).sum(1)
    return loss


def nll_loss_multimodes_joint(pred, ego_data, agents_data, mode_probs, entropy_weight=1.0, kl_weight=1.0,
                              use_FDEADE_aux_loss=True, agent_types=None, predict_yaw=False):
    """
    Args:
      pred: [c, T, B, M, 5]
      ego_data: [B, T, 5]
      agents_data: [B, T, M, 5]
      mode_probs: [B, c], prior prob over modes
    """
    gt_agents = torch.cat((ego_data.unsqueeze(2), agents_data), dim=2)
    modes = len(pred)
    nSteps, batch_sz, N, dim = pred[0].shape
    agents_masks = torch.cat((torch.ones(batch_sz, nSteps, 1).to(ego_data.device), agents_data[:, :, :, -1]), dim=-1)

    # compute posterior probability based on predicted prior and likelihood of predicted scene.
    log_lik = np.zeros((batch_sz, modes))
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist_joint(pred[kk].transpose(0, 1), gt_agents[:, :, :, :2], agents_masks)
            log_lik[:, kk] = -nll.cpu().numpy()

    priors = mode_probs.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors)
    log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=1).reshape((batch_sz, 1))
    post_pr = np.exp(log_posterior)
    post_pr = torch.tensor(post_pr).float().to(gt_agents.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()

    # Compute loss.
    loss = 0.0
    for kk in range(modes):
        nll_k = nll_pytorch_dist_joint(pred[kk].transpose(0, 1), gt_agents[:, :, :, :2], agents_masks) * post_pr[:, kk]
        loss += nll_k.mean()

    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions_joint(pred[kk]).entropy())
    entropy_loss = torch.mean(torch.stack(entropy_vals).permute(2, 0, 3, 1).sum(3).mean(2).max(1)[0])
    loss += entropy_weight * entropy_loss

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
    kl_loss = kl_weight*kl_loss_fn(torch.log(mode_probs), post_pr)

    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    if use_FDEADE_aux_loss:
        adefde_loss = l2_loss_fde_joint(pred, gt_agents, agents_masks, agent_types, predict_yaw)
    else:
        adefde_loss = torch.tensor(0.0).to(gt_agents.device)

    return loss, kl_loss, post_entropy, adefde_loss


def l2_loss_fde_joint(pred, data, agent_masks, agent_types, predict_yaw):
    fde_loss = (torch.norm((pred[:, -1, :, :, :2].transpose(0, 1) - data[:, -1, :, :2].unsqueeze(1)), 2, dim=-1) *
                agent_masks[:, -1:, :]).mean(-1)
    ade_loss = (torch.norm((pred[:, :, :, :, :2].transpose(1, 2) - data[:, :, :, :2].unsqueeze(0)), 2, dim=-1) *
                agent_masks.unsqueeze(0)).mean(-1).mean(dim=2).transpose(0, 1)

    yaw_loss = torch.tensor(0.0).to(pred.device)
    if predict_yaw:
        vehicles_only = (agent_types[:, :, 0] == 1.0).unsqueeze(0)
        yaw_loss = torch.norm(pred[:, :, :, :, 5:].transpose(1, 2) - data[:, :, :, 4:5].unsqueeze(0), dim=-1).mean(2)  # across time
        yaw_loss = (yaw_loss * vehicles_only).mean(-1).transpose(0, 1)

    loss, min_inds = (fde_loss + ade_loss + yaw_loss).min(dim=1)
    return 100.0 * loss.mean()
