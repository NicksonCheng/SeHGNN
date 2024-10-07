import os
import sys
import gc
import random

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append("../data")
from data_loader import data_loader

import warnings

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def visualization(ratio_embs, ratio_labels, save_file, display=False):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    label_rate = ["20", "40", "60"]
    ratio_embs = [embs.cpu().detach().numpy() for embs in ratio_embs]
    ratio_labels = [labels.cpu().detach().numpy() for labels in ratio_labels]
    perplexity = min(30, ratio_embs[0].shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    ratio_embs_2d = [tsne.fit_transform(embs) for embs in ratio_embs]
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "pink", "brown", "gray"]
    if display:
        fig, axs = plt.subplots(1, len(ratio_embs), figsize=(36, 8))
        fig.suptitle("t-SNE visualization of node embeddings with class labels")
        for i, (embs_2d, labels) in enumerate(zip(ratio_embs_2d, ratio_labels)):

            for label in np.unique(labels):
                indices = [i for i, lbl in enumerate(labels) if lbl == label]
                axs[i].scatter(embs_2d[indices, 0], embs_2d[indices, 1], color=colors[label], label=f"Class {label}", alpha=0.6)

            axs[i].set_title(f"{label_rate[i]}Train label node per class")
            axs[i].set_xlabel("x t-SNE vector")
            axs[i].set_ylabel("y t-SNE vector")
            axs[i].legend()

        fig.savefig(save_file)
    ratio_embs_2d = [torch.tensor(embs_2d) for embs_2d in ratio_embs_2d]
    return ratio_embs_2d


def evaluator(pred_score, gt, pred,embeds, num_classes, multilabel=False):
    from sklearn.metrics import roc_auc_score
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    print(gt.shape, pred.shape)
    softmax_score = F.softmax(pred_score.to(torch.float32), dim=1).cpu().detach().numpy()
    pred_score = pred_score.to(torch.float32).cpu().detach().numpy()
    embeds=embeds.to(torch.float32).cpu().detach().numpy()
    # accuracy= (pred == gt).sum().item() / len(gt)
    auc_score = roc_auc_score(
        y_true=gt.cpu().detach().numpy(),
        y_score=softmax_score,
        multi_class="ovr",
        average=None if multilabel else "macro",
    )
    nmi_list = []
    ari_list = []
    for kmeans_random_state in range(10):
        Y_pred = KMeans(n_clusters=num_classes, random_state=kmeans_random_state, n_init=10).fit(embeds).predict(embeds)
        nmi = normalized_mutual_info_score(gt, Y_pred)
        ari = adjusted_rand_score(gt, Y_pred)
        nmi_list.append(nmi)
        ari_list.append(ari)
    return f1_score(gt, pred, average="micro"), f1_score(gt, pred, average="macro"), auc_score, np.mean(nmi_list), np.mean(ari_list)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def hg_propagate_feat_dgl(g, tgt_type, num_hops, max_length, extra_metapath, echo=False):
    for hop in range(1, max_length):
        reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            # if hop == args.num_hops and dtype != tgt_type: continue
            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f"{dtype}{k}"
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo:
                        print(k, etype, current_dst_name)
                    g[etype].update_all(fn.copy_u(k, "m"), fn.mean("m", current_dst_name), etype=etype)

        # remove no-use items
        for ntype in g.ntypes:
            if ntype == tgt_type:
                continue
            removes = []
            for k in g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            if echo and len(removes):
                print("remove", removes)
        gc.collect()

        if echo:
            print(f"-- hop={hop} ---")
        for ntype in g.ntypes:
            for k, v in g.nodes[ntype].data.items():
                print(f"{ntype} {k} {v.shape}", v[:, -1].max(), v[:, -1].mean())
        if echo:
            print(f"------\n")
    return g


def hg_propagate_sparse_pyg(adjs, tgt_types, num_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device="cpu"):
    store_device = "cpu"
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {
        k: v.clone() for k, v in adjs.items() if prop_feats or k[-1] in tgt_types
    }  # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop + 1) :] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items():
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]
                for rtype_l, adj_l in adjs_g.items():
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f"{dtype_l}{rtype_r}"
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) or (
                            hop > num_hops and name not in reserve_heads
                        ):
                            continue
                        if name not in new_adjs:
                            if echo:
                                print("Generating ...", name)
                            if prop_device == "cpu":

                                new_adjs[name] = adj_l.matmul(adj_r)
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                        else:
                            if echo:
                                print(f"Warning: {name} already exists")
        label_feats.update(new_adjs)

        removes = []
        for k in label_feats.keys():
            metapath_types = list(k)
            if metapath_types[0] in tgt_types:
                continue  # metapath should end with target type in label propagation
            if len(metapath_types) <= hop:
                removes.append(k)
        for k in removes:
            label_feats.pop(k)
        if echo and len(removes):
            print("remove", removes)
        del new_adjs
        gc.collect()

    if prop_device != "cpu":
        del adjs_g
        torch.cuda.empty_cache()

    return label_feats


def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid, show_test=True, loss_type="ce"):

    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    k = list(preds_dict.keys())[0]
    v = preds_dict[k]
    if loss_type == "ce":
        na, nb, nc = len(train_nid), len(val_nid), len(test_nid)
    elif loss_type == "bce":
        na, nb, nc = len(train_nid) * v.size(1), len(val_nid) * v.size(1), len(test_nid) * v.size(1)

    for k, v in preds_dict.items():
        if loss_type == "ce":
            pred = v.argmax(1)
        elif loss_type == "bce":
            pred = (v > 0).int()
        a, b, c = pred[train_nid] == init_labels[train_nid], pred[val_nid] == init_labels[val_nid], pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / na, b.sum() / nb, c.sum() / nc

        if loss_type == "ce":
            vv = torch.log(v / (v.sum(1, keepdim=True) + 1e-6) + 1e-6)
            la, lb, lc = (
                F.nll_loss(vv[train_nid], init_labels[train_nid]),
                F.nll_loss(vv[val_nid], init_labels[val_nid]),
                F.nll_loss(vv[test_nid], init_labels[test_nid]),
            )
        else:
            vv = (v / 2.0 + 0.5).clamp(1e-6, 1 - 1e-6)
            la, lb, lc = (
                F.binary_cross_entropy(vv[train_nid], init_labels[train_nid].float()),
                F.binary_cross_entropy(vv[val_nid], init_labels[val_nid].float()),
                F.binary_cross_entropy(vv[test_nid], init_labels[test_nid].float()),
            )
        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        if show_test:
            print(k, ra, rb, rc, la, lb, lc, (ra / rb - 1) * 100, (ra / rc - 1) * 100, (1 - la / lb) * 100, (1 - la / lc) * 100)
        else:
            print(k, ra, rb, la, lb, (ra / rb - 1) * 100, (1 - la / lb) * 100)
    print(set(list(preds_dict.keys())) - set(remove_label_keys))

    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / na)
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / nb)
    if show_test:
        print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / nc)


def train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, num_classes, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred,y_embeds = [], [], []
    y_pred_score = []
    for batch in train_loader:
        # batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att,embeds = model(batch, batch_feats, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att,embeds = model(batch, batch_feats, batch_labels_feats, batch_mask)
            loss_train = loss_fcn(output_att, batch_y)
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.0).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        y_pred_score.append(output_att)
        y_embeds.append(embeds)
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    # print("Training", output_att.shape, torch.cat(y_true, dim=0).shape, torch.cat(y_pred, dim=0).shape)
    acc = evaluator(torch.cat(y_pred_score, dim=0), torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0),torch.cat(y_embeds,dim=0), num_classes)
    return loss, acc


def _read_edges(raw_path, ntypes):
    import pandas as pd

    edges = {}
    feats = {}
    for file in os.listdir(raw_path):
        name, ext = os.path.splitext(file)
        if "_feat" in name:
            file = os.path.join(raw_path, f"{name}.npz")
            u = name.split("_")[0]
            feats[u] = torch.from_numpy(sp.load_npz(file).toarray()).float()
        if ext == ".txt":
            u, v = name
            v_name = f"{v}_" if u == v else v
            e = pd.read_csv(os.path.join(raw_path, f"{u}{v}.txt"), sep="\t", names=[u, v_name])
            src = e[u].to_list()
            dst = e[v].to_list()
            # src_name, dst_name = ntypes[u], ntypes[v]
            src_name, dst_name = u, v
            edges[(src_name, f"{src_name}-{dst_name}", dst_name)] = (src, dst)
            edges[(dst_name, f"{dst_name}-{src_name}", src_name)] = (dst, src)
    g = dgl.heterograph(edges)
    return g, feats


def load_labels_with_ratio(raw_path, g, predict_ntype):
    from dgl.data.utils import (
        generate_mask_tensor,
        idx2mask,
    )

    label_ratio = ["20", "40", "60"]
    labels = torch.from_numpy(np.load(os.path.join(raw_path, "labels.npy"))).long()
    num_classes = labels.max().item() + 1
    n = g.num_nodes(predict_ntype)
    ratio_init_labels = {ratio: np.zeros(n, dtype=int) for ratio in label_ratio}
    ratio_nid = {ratio: {} for ratio in label_ratio}
    class_nid = {id: [] for id in range(num_classes)}
    for idx, lb in enumerate(labels):
        class_nid[lb.item()].append(idx)
    for ratio in label_ratio:
        idx_train = np.load(os.path.join(raw_path, f"train_{ratio}.npy"))
        idx_val = np.load(os.path.join(raw_path, f"val_{ratio}.npy"))
        idx_test = np.load(os.path.join(raw_path, f"test_{ratio}.npy"))
        ratio_init_labels[ratio][idx_train] = labels[idx_train]
        ratio_init_labels[ratio][idx_val] = labels[idx_val]
        ratio_init_labels[ratio][idx_test] = labels[idx_test]

        ratio_init_labels[ratio] = torch.LongTensor(ratio_init_labels[ratio])
        ratio_nid[ratio]["train"] = idx_train
        ratio_nid[ratio]["val"] = idx_val
        ratio_nid[ratio]["test"] = idx_test

    return num_classes, ratio_init_labels, ratio_nid


def load_labels(raw_path, g, predict_ntype):

    ## 要 5個 train/test idx 去訓練
    from sklearn.model_selection import StratifiedKFold

    n_splits = 10
    labels = torch.from_numpy(np.load(os.path.join(raw_path, "labels.npy"))).long()
    num_classes = labels.max().item() + 1
    n = g.num_nodes(predict_ntype)
    ratio_init_labels = {i: np.zeros(n, dtype=int) for i in range(n_splits)}
    ratio_nids = {i: [] for i in range(n_splits)}
    seed = np.random.seed(1)

    X = np.arange(len(labels)).reshape(-1, 1)  # dummy X
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (trainval_idx, test_idx) in enumerate(skf.split(X=X, y=labels)):
        label_train_val = labels[trainval_idx]
        skf_train_val = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf_train_val.split(X=X[trainval_idx], y=label_train_val):
            train_idx = trainval_idx[train_idx]  # Adjust indices to original dataset
            val_idx = trainval_idx[val_idx]  # Adjust indices to original dataset

            train_idx = torch.from_numpy(train_idx)
            val_idx = torch.from_numpy(val_idx)
            test_idx = torch.from_numpy(test_idx)
            ratio_init_labels[i][train_idx] = labels[train_idx]
            ratio_init_labels[i][val_idx] = labels[val_idx]
            ratio_init_labels[i][test_idx] = labels[test_idx]
            ratio_init_labels[i] = torch.LongTensor(ratio_init_labels[i])
            # print(len(trainval_idx), len(train_idx), len(val_idx), len(test_idx))
            ratio_nids[i] = {"train": train_idx, "val": val_idx, "test": test_idx}
            break
    return num_classes, ratio_init_labels, ratio_nids


def load_dataset(args):
    import torch.nn.functional as F

    large_dataset = ["Freebase"]
    if args.dataset not in large_dataset:
        dataset_ntypes = {
            "acm": {"p": "paper", "a": "author", "s": "subject"},
            "aminer": {"p": "paper", "a": "author", "r": "reference"},
            "freebase": {"m": "movie", "a": "author", "w": "writer", "d": "director"},
            "PubMed": {"G": "Geng", "D": "Disease", "C": "Chemical", "S": "Species"},
        }
        dataset_predict = {"acm": "p", "aminer": "p", "freebase": "m", "PubMed": "D"}
        raw_path = f"{args.root}/{args.dataset}" if args.dataset == "PubMed" else f"{args.root}/HeCo/{args.dataset}"
        hg, feats = _read_edges(raw_path, dataset_ntypes[args.dataset])
        features_list = []

        for ntype in dataset_ntypes[args.dataset].keys():
            if ntype in feats:
                hg.nodes[ntype].data[ntype] = feats[ntype]
                features_list.append(feats[ntype])
            else:
                hg.nodes[ntype].data[ntype] = torch.eye(hg.num_nodes(ntype))
                features_list.append(torch.eye(hg.num_nodes(ntype)))
        max_feat_dim = max([x.shape[1] for x in features_list])
        for ntype in dataset_ntypes[args.dataset].keys():
            feat = hg.nodes[ntype].data[ntype]
            if feat.shape[1] < max_feat_dim:
                pad = F.pad(feat, (0, max_feat_dim - feat.shape[1]))
                hg.nodes[ntype].data[ntype] = pad

        adjs = {}
        for etype in hg.canonical_etypes:

            src, dst = hg.edges(etype=etype)
            num_src, num_dst = hg.num_nodes(etype[0]), hg.num_nodes(etype[2])
            adj = SparseTensor(row=torch.LongTensor(src), col=torch.LongTensor(dst), sparse_sizes=(num_src, num_dst))
            name = f"{etype[0]}{etype[2]}"
            adjs[name] = adj
        if args.dataset == "PubMed":
            num_classes, ratio_init_labels, ratio_nid = load_labels(raw_path, hg, dataset_predict[args.dataset])
        else:
            num_classes, ratio_init_labels, ratio_nid = load_labels_with_ratio(raw_path, hg, dataset_predict[args.dataset])
        return hg, adjs, ratio_init_labels, num_classes, ratio_nid

    dl = data_loader(f"{args.root}/{args.dataset}")
    # use one-hot index vectors for nods with no attributes
    # === feats ===
    features_list = []
    for i in range(len(dl.nodes["count"])):
        th = dl.nodes["attr"][i]

        if th is None:
            features_list.append(torch.eye(dl.nodes["count"][i]))
        else:
            features_list.append(torch.FloatTensor(th))
    idx_shift = np.zeros(len(dl.nodes["count"]) + 1, dtype=np.int32)
    for i in range(len(dl.nodes["count"])):
        idx_shift[i + 1] = idx_shift[i] + dl.nodes["count"][i]

    # === labels ===
    num_classes = dl.labels_train["num_classes"]
    init_labels = np.zeros((dl.nodes["count"][1], num_classes), dtype=int)
    trainval_nid = np.nonzero(dl.labels_train["mask"])[0]
    test_nid = np.nonzero(dl.labels_test["mask"])[0]

    ## because node type is 1, we need to shift node label indcies
    init_labels[trainval_nid] = dl.labels_train["data"][trainval_nid]
    init_labels[test_nid] = dl.labels_test["data"][test_nid]
    if args.dataset != "IMDB":
        init_labels = init_labels.argmax(axis=1)
    init_labels = torch.LongTensor(init_labels)
    # === adjs ===
    # print(dl.nodes['attr'])
    # for k, v in dl.nodes['attr'].items():
    #     if v is None: print('none')
    #     else: print(v.shape)
    adjs = [] if args.dataset != "Freebase" else {}
    for i, (k, v) in enumerate(dl.links["data"].items()):
        v = v.tocoo()
        src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
        dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
        row = v.row - idx_shift[dst_type_idx]
        col = v.col - idx_shift[src_type_idx]
        sparse_sizes = (dl.nodes["count"][dst_type_idx], dl.nodes["count"][src_type_idx])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        if args.dataset == "Freebase":
            name = f"{dst_type_idx}{src_type_idx}"
            assert name not in adjs
            adjs[name] = adj
        else:
            adjs.append(adj)
            print(adj)
    print(adjs.keys())
    exit()
    if args.dataset == "DBLP":
        # A* --- P --- T
        #        |
        #        V
        # author: [4057, 334]
        # paper : [14328, 4231]
        # term  : [7723, 50]
        # venue(conference) : None
        A, P, T, V = features_list
        AP, PA, PT, PV, TP, VP = adjs

        new_edges = {}
        ntypes = set()
        etypes = [  # src->tgt
            ("P", "P-A", "A"),
            ("A", "A-P", "P"),
            ("T", "T-P", "P"),
            ("V", "V-P", "P"),
            ("P", "P-T", "T"),
            ("P", "P-V", "V"),
        ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        # for i, etype in enumerate(g.etypes):
        #     src, dst, eid = g._graph.edges(i)
        #     adj = SparseTensor(row=dst.long(), col=src.long())
        #     print(etype, adj)

        # g.ndata['feat']['A'] = A # not work
        g.nodes["A"].data["A"] = A
        g.nodes["P"].data["P"] = P
        g.nodes["T"].data["T"] = T
        g.nodes["V"].data["V"] = V
    elif args.dataset == "IMDB":
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        M, D, A, K = features_list
        MD, DM, MA, AM, MK, KM = adjs
        assert torch.all(DM.storage.col() == MD.t().storage.col())
        assert torch.all(AM.storage.col() == MA.t().storage.col())
        assert torch.all(KM.storage.col() == MK.t().storage.col())

        assert torch.all(MD.storage.rowcount() == 1)  # each movie has single director

        new_edges = {}
        ntypes = set()
        etypes = [  # src->tgt
            ("D", "D-M", "M"),
            ("M", "M-D", "D"),
            ("A", "A-M", "M"),
            ("M", "M-A", "A"),
            ("K", "K-M", "M"),
            ("M", "M-K", "K"),
        ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        g.nodes["M"].data["M"] = M
        g.nodes["D"].data["D"] = D
        g.nodes["A"].data["A"] = A
        if args.num_hops > 2 or args.two_layer:
            g.nodes["K"].data["K"] = K
    elif args.dataset == "ACM":
        # A --- P* --- C
        #       |
        #       K
        # paper     : [3025, 1902]
        # author    : [5959, 1902]
        # conference: [56, 1902]
        # field     : None
        # P,A,S=features_list
        P, A, C, K = features_list
        PP, PP_r, PA, AP, PC, CP, PK, KP = adjs
        row, col = torch.where(P)
        assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
        assert torch.all(AP.matmul(PK).to_dense() == A)
        assert torch.all(CP.matmul(PK).to_dense() == C)

        assert torch.all(PA.storage.col() == AP.t().storage.col())
        assert torch.all(PC.storage.col() == CP.t().storage.col())
        assert torch.all(PK.storage.col() == KP.t().storage.col())

        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())
        PP = PP.coalesce()
        PP = PP.set_diag()
        adjs = [PP] + adjs[2:]

        new_edges = {}
        ntypes = set()
        etypes = [  # src->tgt
            ("P", "P-P", "P"),
            ("A", "A-P", "P"),
            ("P", "P-A", "A"),
            ("C", "C-P", "P"),
            ("P", "P-C", "C"),
        ]
        if args.ACM_keep_F:
            etypes += [
                ("K", "K-P", "P"),
                ("P", "P-K", "K"),
            ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)

        g = dgl.heterograph(new_edges)

        g.nodes["P"].data["P"] = P  # [3025, 1902]
        g.nodes["A"].data["A"] = A  # [5959, 1902]
        g.nodes["C"].data["C"] = C  # [56, 1902]
        if args.ACM_keep_F:
            g.nodes["K"].data["K"] = K  # [1902, 1902]
    elif args.dataset == "Freebase":
        # 0*: 40402  2/4/7 <-- 0 <-- 0/1/3/5/6
        #  1: 19427  all <-- 1
        #  2: 82351  4/6/7 <-- 2 <-- 0/1/2/3/5
        #  3: 1025   0/2/4/6/7 <-- 3 <-- 1/3/5
        #  4: 17641  4 <-- all
        #  5: 9368   0/2/3/4/6/7 <-- 5 <-- 1/5
        #  6: 2731   0/4 <-- 6 <-- 1/2/3/5/6/7
        #  7: 7153   4/6 <-- 7 <-- 0/1/2/3/5/7
        for i in range(8):
            kk = str(i)
            print(f"==={kk}===")
            for k, v in adjs.items():
                t, s = k
                assert s == t or f"{s}{t}" not in adjs
                if s == kk or t == kk:
                    if s == t:
                        print(k, v.sizes(), v.nnz(), f"symmetric {v.is_symmetric()}; selfloop-ratio: {v.get_diag().sum()}/{v.size(0)}")
                    else:
                        print(k, v.sizes(), v.nnz())

        adjs["00"] = adjs["00"].to_symmetric()
        g = None
    else:
        assert 0

    if args.dataset == "DBLP":
        adjs = {"AP": AP, "PA": PA, "PT": PT, "PV": PV, "TP": TP, "VP": VP}
    elif args.dataset == "ACM":
        adjs = {"PP": PP, "PA": PA, "AP": AP, "PC": PC, "CP": CP}
    elif args.dataset == "IMDB":
        adjs = {"MD": MD, "DM": DM, "MA": MA, "AM": AM, "MK": MK, "KM": KM}
    elif args.dataset == "Freebase":
        new_adjs = {}
        for rtype, adj in adjs.items():
            dtype, stype = rtype
            if dtype != stype:
                new_name = f"{stype}{dtype}"
                assert new_name not in adjs
                new_adjs[new_name] = adj.t()
        adjs.update(new_adjs)
    else:
        assert 0

    return g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid


class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, save_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
