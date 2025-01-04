import os, torch, tqdm, random, yaml
import numpy as np
from utils.dataset import *
from utils.llm import *
from utils.encoder import *
from utils.physiclear_constants import get_categorical_labels
from datetime import datetime
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics


def visualize(configs, loaders, models, split, pca, device, exp_id, train, test):
    models["tactile_vificlip"].eval()
    if not configs["prompt_learning"]:
        models["tactile_adapter"].eval()
    models["plain_tactile_adapter"].eval()
    models["dotted_tactile_adapter"].eval()
    models["property_classifier"].eval()
    num_prop_cls_samples = 0
    if "property_regression" in configs["tasks"]:
        prop_reg_loader = loaders["property_regression"]
    all_embeddings, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(prop_reg_loader):
            if "property_regression" in configs["tasks"]:
                # Task 1: Property classification
                all_tactile_frames, properties, dataset = batch
                all_labels.append(properties.cpu().numpy())
                batch_size = all_tactile_frames.shape[0]
                num_prop_cls_samples += batch_size
                # 1.1: Tactile
                tactile_video_features, _, _, _ = models["tactile_vificlip"](all_tactile_frames.to(device), None, None)
                if not configs["prompt_learning"]:
                    tactile_video_features = models["tactile_adapter"](tactile_video_features)
                tactile_video_features = models["dotted_tactile_adapter"](tactile_video_features)
                plain_indices = [i for i, x in enumerate(dataset) if get_dataset_sensor_type(x) == "plain"]
                plain_tactile_video_features = models["plain_tactile_adapter"](tactile_video_features)
                tactile_video_features_clone = tactile_video_features.clone()
                tactile_video_features_clone[plain_indices] = plain_tactile_video_features[plain_indices]
                # 1.2: Regression
                prop_preds = models["property_classifier"](tactile_video_features_clone)
                all_preds.append(prop_preds.cpu().numpy())
                # 1.3: Embeddings
                all_embeddings.append(tactile_video_features_clone.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_labels_bin = []
    for l in all_labels:
        all_labels_bin.append(np.asarray([get_categorical_labels(l[0], bins=configs["visualize_bins"]), get_categorical_labels(l[1], bins=configs["visualize_bins"])]))
    all_labels_bin = np.concatenate([all_labels_bin], axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_preds_bin = []
    for p in all_preds:
        all_preds_bin.append(np.asarray([get_categorical_labels(p[0], bins=configs["visualize_bins"]), get_categorical_labels(p[1], bins=configs["visualize_bins"])]))
    all_preds_bin = np.concatenate([all_preds_bin], axis=0)
    if not train and test:
        # 1) Classification
        num_samples = all_preds_bin.shape[0]
        hardness_acc = np.sum(all_preds_bin[:, 0] == all_labels_bin[:, 0]) / num_samples
        roughness_acc = np.sum(all_preds_bin[:, 1] == all_labels_bin[:, 1]) / num_samples
        combined_acc = np.sum(np.all(all_preds_bin == all_labels_bin, axis=-1)) / num_samples
        results = {
            "Hardness": hardness_acc,
            "Roughness": roughness_acc,
            "Combined": combined_acc
        }
        acc_json_path = f'{configs["exps_path"]}/{exp_id}/results/encoder_cls_{split}.json'
        with open(acc_json_path, 'w') as f:
            json.dump(results, f, indent=4)
            f.close()
    # 2) Confusion matrix
    # hardness_order = ["Soft", "Semi-soft", "Semi-hard", "Hard"]
    labels = [i for i in range(configs["visualize_bins"])]
    hardness_order = [i for i in range(configs["visualize_bins"])]
    hardness_confusion_matrix = metrics.confusion_matrix(all_labels_bin[:, 0], all_preds_bin[:, 0], labels=labels)
    hardness_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=hardness_confusion_matrix, display_labels=hardness_order)
    hardness_cm_display.plot()
    plt.savefig(f"{configs['exps_path']}/{exp_id}/results/confusion_matrix_hardness.png")
    plt.clf()
    # roughness_order = ["Smooth", "Semi-smooth", "Semi-rough", "Rough"]
    roughness_order = [i for i in range(configs["visualize_bins"])]
    roughness_confusion_matrix = metrics.confusion_matrix(all_labels_bin[:, 1], all_preds_bin[:, 1], labels=labels)
    rougness_cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=roughness_confusion_matrix, display_labels=roughness_order)
    rougness_cm_display.plot()
    plt.savefig(f"{configs['exps_path']}/{exp_id}/results/confusion_matrix_roughness.png")
    plt.clf()
    # 3) Embeddings
    # PCA
    df = pd.DataFrame()
    # hardness_terms = {0: 'Soft', 1: 'Semi-soft', 2: 'Semi-hard', 3: 'Hard'}
    # df['Hardness Labels'] = [hardness_terms[int(i)] for i in all_labels_bin[:,0]]
    df['Hardness Labels'] = [int(i) for i in all_labels_bin[:,0]]
    # roughness_terms = {0: 'Smooth', 1: 'Semi-smooth', 2: 'Semi-rough', 3: 'Rough'}
    # df['Roughness Labels'] = [roughness_terms[int(i)] for i in all_labels_bin[:,1]]
    df['Roughness Labels'] = [int(i) for i in all_labels_bin[:,1]]
    titles = {0: "hardness", 1: "roughness"}
    orders = {0: hardness_order, 1: roughness_order}
    labels_name = {0: "Hardness Labels", 1: "Roughness Labels"}
    labels_num = {0: configs["visualize_bins"], 1: configs["visualize_bins"]}
    if pca is None:
        pca = PCA(n_components=30)
        pca.fit(all_embeddings)
    pca_result = pca.transform(all_embeddings)
    print('Cumulative explained variation for the principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    # t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result)
    df["PCA t-SNE 1"] = tsne_results[:,0]
    df["PCA t-SNE 2"] = tsne_results[:,1]
    for label_type_idx in range(all_labels_bin.shape[1]):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x="PCA t-SNE 1", y="PCA t-SNE 2",
            hue_order=orders[label_type_idx],
            hue=labels_name[label_type_idx],
            palette=sns.color_palette("hls", labels_num[label_type_idx]),
            data=df,
            legend="full",
        )
        plt.xlabel("PCA t-SNE 1", fontsize=18)
        plt.ylabel("PCA t-SNE 2", fontsize=18)
        plt.legend(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.savefig(f"{configs['exps_path']}/{exp_id}/viz/tsne_{split}_{titles[label_type_idx]}.png")
        plt.clf()