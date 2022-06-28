import ast
from copy import deepcopy
import json
import numpy as np


def group_uniques_full(hist, losses_to_average, verbose=False, group_norm_diffs=False):
    grouped_hist = {}
    unique_tried = {}
    for one_optim_hist in hist:
        unique_tried[one_optim_hist["name"]] = False

    for one_optim_hist in hist:
        label = one_optim_hist["name"]
        if not unique_tried[label]:
            unique_tried[label] = True
            grouped_hist[label] = {
                "hist": deepcopy(one_optim_hist),
                "repeats": {
                    loss_name: [1] * len(one_optim_hist[loss_name]) for loss_name in losses_to_average
                }
            }
            if group_norm_diffs:
                grouped_hist[label]["hist"]["norm_diffs"] = [
                    [np.array(x)] for x in grouped_hist[label]["hist"]["norm_diffs"]
                ]
            continue
        
        for loss_name in losses_to_average:
            losses = one_optim_hist[loss_name]
                
            for i, loss_elem in enumerate(losses):
                if i < len(grouped_hist[label]["hist"][loss_name]):
                    grouped_hist[label]["hist"][loss_name][i] += loss_elem
                else:
                    grouped_hist[label]["hist"][loss_name].append(loss_elem)

                if i >= len(grouped_hist[label]["repeats"][loss_name]):
                    grouped_hist[label]["repeats"][loss_name].append(0)
                grouped_hist[label]["repeats"][loss_name][i] += 1
        
        if group_norm_diffs:
            if len(grouped_hist[label]["hist"]["norm_diffs"]) == 0:
                if "norm_diffs_x" in one_optim_hist:
                    grouped_hist[label]["hist"]["norm_diffs_x"] = one_optim_hist["norm_diffs_x"]
                grouped_hist[label]["hist"]["norm_diffs"] = [
                    [np.array(x)] for x in one_optim_hist["norm_diffs"]
                ]
            else:
                for x, y in zip(grouped_hist[label]["hist"]["norm_diffs"], one_optim_hist["norm_diffs"]):
                    x.append(np.array(y))

    for key in grouped_hist:
        one_optim_hist = grouped_hist[key]
        if verbose and len(one_optim_hist["repeats"][losses_to_average[0]]) > 0:
            repeats_1 = float(one_optim_hist["repeats"][losses_to_average[0]][0])
            print("Repeats_1 = {}, Name = {}".format(repeats_1, one_optim_hist["hist"]["name"]))
        for loss_name in losses_to_average:
            for i in range(len(one_optim_hist["hist"][loss_name])):
                repeats = one_optim_hist["repeats"][loss_name][i]
                one_optim_hist["hist"][loss_name][i] /= repeats
        
        if group_norm_diffs:
            for i, group in enumerate(one_optim_hist["hist"]["norm_diffs"]):
                means = []
                stds = []
                for j, elem in enumerate(group):
                    means.append(elem.mean())
                    stds.append(elem.std())
                    group[j] = (elem - elem.mean()) / elem.std()
                mean = np.mean(means)
                std = np.mean(stds)
                one_optim_hist["hist"]["norm_diffs"][i] = np.concatenate(group) * std + mean

    grouped_hist = [grouped_hist[x]["hist"] for x in grouped_hist]
            
    return grouped_hist


def load_hist_jsons(hists_names_list, path="./models"):
    hists = []
    for hist_name in hists_names_list:
        with open(r"{}/{}.json".format(path, hist_name), "r") as read_file:
            hist = json.load(read_file)
            hists += hist
    return hists


def rec_hist_from_json(h, key):
    for i in range(len(h)):
        if key == "val_norm_diffs":
            h[i] = ast.literal_eval(h[i])
            for j in range(len(h[i])):
                h[i][j] = float(h[i][j])
        elif isinstance(h[i], list):
            rec_hist_from_json(h[i], key)
        else:
            h[i] = float(h[i])


def hist_from_json(hists):
    for h in hists:
        for key in h:
            if isinstance(h[key], list):
                rec_hist_from_json(h[key], key)
    return hists
