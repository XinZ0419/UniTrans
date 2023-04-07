import numpy as np
import torch
import torch.nn.functional as F
from concordance import concordance_index


def evaluate(opt, encoder, test_loader):
    encoder.eval()
    with torch.no_grad():
        pred_durations, true_durations, is_observed = [], [], []
        pred_obs_durations, true_obs_durations = [], []

        total_surv_probs = []
        total_sftmax_probs = []
        is_observed_list = []
        duration_list = []

        # NOTE batch size is 1
        for features, images, durations, mask, label, is_observed_single in test_loader:
            is_observed.append(is_observed_single.cpu())
            output = encoder.forward(features, images)
            softmax_preds = F.softmax(output.squeeze(-1), dim=-1)
            surv_probs = (1 - torch.cumsum(softmax_preds, dim=1)).squeeze()
            total_surv_probs.append(surv_probs)
            total_sftmax_probs.append(softmax_preds.squeeze())
            is_observed_list.append(is_observed_single)
            duration_list.append(durations)

            if opt.pred_method == 'mean':
                pred_duration = torch.sum(surv_probs).item()
            elif opt.pred_method == 'median':
                pred_duration = 0
                while True:
                    if surv_probs[pred_duration] < 0.5:
                        break
                    else:
                        pred_duration += 1
                        if pred_duration == len(surv_probs):
                            break

            true_duration = durations.squeeze().item()
            pred_durations.append(pred_duration)
            true_durations.append(true_duration)

            if is_observed_single:
                pred_obs_durations.append(pred_duration)
                true_obs_durations.append(true_duration)

        total_surv_probs = torch.stack(total_surv_probs)
        total_sftmax_probs = torch.stack(total_sftmax_probs)
        is_observed_list = torch.stack(is_observed_list)
        duration_list = torch.stack(duration_list)

        pred_obs_durations = np.asarray(pred_obs_durations)
        true_obs_durations = np.asarray(true_obs_durations)
        mae_obs = np.mean(np.abs(pred_obs_durations - true_obs_durations))

        pred_durations = np.asarray(pred_durations)
        true_durations = np.asarray(true_durations)
        is_observed = np.asarray(is_observed, dtype=bool)

        print('pred durations OBS', pred_durations[is_observed].round())
        print('true durations OBS', true_durations[is_observed].round())

        print('pred durations CRS', pred_durations[~is_observed].round())
        print('true durations CRS', true_durations[~is_observed].round())

        test_cindex = concordance_index(true_durations, pred_durations, is_observed)
        print('c index', test_cindex, 'mean abs error (OBS)', mae_obs)

    return test_cindex, mae_obs, total_surv_probs, total_sftmax_probs, is_observed_list, duration_list
