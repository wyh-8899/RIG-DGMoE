# -*- coding: utf-8 -*-
"""
Reference-aligned dataloader (COVER only, HEAD crop only) with simple cache/log style.

Outputs:
- TrafficState: [B, Th, N, 1]
- FutureState : [B, Tf, N, 1]
- Trajectory: [B, Th, M, L, 3]
- FutureTraj: [B, Tf, M, L, 3]
- Et          : [B, Th+Tf, 2]   (tod, dow)

Masks:
- Trajectory_car_mask   [B, Th, M]
- FutureTraj_car_mask   [B, Tf, M]
- Trajectory_token_mask [B, Th, M, L]
- FutureTraj_token_mask [B, Tf, M, L]
- Trajectory_step_mask  [B, Th, M, L-1]
- FutureTraj_step_mask  [B, Tf, M, L-1]
"""
from torch.utils.data.distributed import DistributedSampler
import os
import ast
import pickle
import random
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------
# logger (reference-like)
# --------------------------
def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logger


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


def crop_head(path: List[int], L: int) -> List[int]:
    return path if len(path) <= L else path[:L]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# --------------------------
# Dataset (cover-only)
# --------------------------
class STREDataset(Dataset):
    def __init__(self, config: Dict, phase: str = "train"):
        self.logger = get_logger(f"data_loader.{phase}")
        self.phase = phase

        # ----- config -----
        self.dataset = config.get("dataset", "xa")
        self.data_root = "/data/LvHaochen/Dachuang-wyh/TRACK-master/raw_data/"
        self.data_path = os.path.join(self.data_root, self.dataset)

        self.time_intervals = int(config.get("time_intervals", 1800))
        self.input_window = int(config.get("input_window", 6))    # Th
        self.output_window = int(config.get("output_window", 1))  # Tf

        self.traj_batch_size = int(config.get("traj_batch_size", config.get("traj_sample_num", 10)))  # M
        self.max_traj_len = int(config.get("max_traj_len", 20))  # L

        self.strict_split_window = bool(config.get("strict_split_window", True))
        self.require_nonempty = bool(config.get("require_nonempty", True))

        self.seed = int(config.get("seed", 42))
        self.rng = random.Random(self.seed + (0 if phase == "train" else 777))

        # degree 不是必须，这里保留 config 字段但本实现不再使用
        self.add_degree = bool(config.get("add_degree", False))

        # ----- cache -----
        self.cache_dir = config.get("cache_dir", os.path.join(self.data_root, "_cache"))
        ensure_dir(self.cache_dir)

        # ----- load or build -----
        self._load_or_build_geo()
        self._load_or_build_geo_feature()   # -> node_features + L_r
        self._load_or_build_rel()
        self._load_or_build_dyna()          # -> traffic_data + time_features + u_ff_r

        self._load_or_build_traj_dict()
        self._load_or_build_traj_batches()

        self._load_or_build_valid_indices()
        self._load_or_build_cover_indices()

        if self.require_nonempty and len(self.sample_indices) == 0:
            raise RuntimeError(f"[{self.phase.upper()}] Dataset empty after cover expansion.")

    # --------------------------
    # GEO
    # --------------------------
    def _geo_cache_file(self):
        return os.path.join(self.cache_dir, f"{self.dataset}_geo.pkl")

    def _load_or_build_geo(self):
        cache_file = self._geo_cache_file()
        if os.path.exists(cache_file):
            obj = pickle.load(open(cache_file, "rb"))
            self.geo_ids = obj["geo_ids"]
            self.geo2ind = obj["geo2ind"]
            self.num_nodes = obj["num_nodes"]
            self.pad_value = obj["pad_value"]
            self.logger.info(f"[{self.phase.upper()}] Load geo cache: {cache_file}")
        else:
            geo_path = os.path.join(self.data_path, f"{self.dataset}.geo")
            df = pd.read_csv(geo_path)
            self.geo_ids = [safe_int(x) for x in df["geo_id"].tolist()]
            self.num_nodes = len(self.geo_ids)
            self.geo2ind = {int(gid): i for i, gid in enumerate(self.geo_ids)}
            self.pad_value = self.num_nodes

            pickle.dump(
                {"geo_ids": self.geo_ids,
                 "geo2ind": self.geo2ind,
                 "num_nodes": self.num_nodes,
                 "pad_value": self.pad_value},
                open(cache_file, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL
            )
            self.logger.info(f"[{self.phase.upper()}] Save geo cache: {cache_file}")

        self.logger.info(f"[{self.phase.upper()}] Loaded {self.num_nodes} nodes. pad_value={self.pad_value}")

    # --------------------------
    # GEO FEATURES + L_r
    # --------------------------
    def _geo_feat_cache_file(self):
        # 这里不再做 degree 分支，因为 geo 中没有 indegree/outdegree；也不是必须
        return os.path.join(self.cache_dir, f"{self.dataset}_node_features.npy")

    def _load_or_build_geo_feature(self):
        """
        Build node_features: float32 [N, F], aligned with geo_ids order (0..N-1).

        Mimic reference (but remove degree):
          useful cols: ['highway','lanes','length','maxspeed']
          - length: min-max for node_features only
          - lanes/maxspeed/highway: one-hot

        Also extract raw L_r from .geo:
          L_r: float32 [N] raw length (no normalization), for ImpedanceNet physical term.
        """
        node_fea_path = self._geo_feat_cache_file()

        # always read geo once to get L_r (raw)
        geo_path = os.path.join(self.data_path, f"{self.dataset}.geo")
        road_info = pd.read_csv(geo_path)
        road_info = road_info.set_index("geo_id").reindex(self.geo_ids).reset_index()

        # L_r: raw length
        if "length" not in road_info.columns:
            raise KeyError(f"{geo_path} missing required column 'length' for L_r.")
        self.L_r = road_info["length"].astype(np.float32).values  # [N]

        if os.path.exists(node_fea_path):
            node_features = np.load(node_fea_path)
            self.logger.info(f"[{self.phase.upper()}] Load geo_feature cache: {node_fea_path}")
        else:
            useful = ["highway", "lanes", "length", "maxspeed"]
            for col in useful:
                if col not in road_info.columns:
                    raise KeyError(f"{geo_path} missing required column '{col}' for node_features.")

            node_df = road_info[useful].copy()

            # 1) length min-max only for node_features
            d = node_df["length"].astype(float)
            min_, max_ = d.min(), d.max()
            denom = (max_ - min_) if (max_ - min_) > 1e-6 else 1.0
            node_df["length"] = (d - min_) / denom

            # 2) one-hot lanes/maxspeed/highway
            onehot_list = ["lanes", "maxspeed", "highway"]
            for col in onehot_list:
                dum = pd.get_dummies(node_df[col], prefix=col)
                node_df = node_df.drop(columns=[col])
                node_df = pd.concat([node_df, dum], axis=1)

            node_features = node_df.values.astype(np.float32)
            np.save(node_fea_path, node_features)
            self.logger.info(f"[{self.phase.upper()}] Save geo_feature cache: {node_fea_path}")

        self.node_features = node_features.astype(np.float32)  # [N, F]
        self.node_fea_dim = int(self.node_features.shape[1])
        self.logger.info(f"[{self.phase.upper()}] node_features: {self.node_features.shape}")
        self.logger.info(f"[{self.phase.upper()}] L_r(raw length): {self.L_r.shape}")

    # --------------------------
    # REL
    # --------------------------
    def _rel_cache_file(self):
        return os.path.join(self.cache_dir, f"{self.dataset}_rel.pkl")

    def _load_or_build_rel(self):
        cache_file = self._rel_cache_file()
        if os.path.exists(cache_file):
            obj = pickle.load(open(cache_file, "rb"))
            self.adj_mx = obj["adj_mx"]
            self.edge_index = obj["edge_index"]
            self.logger.info(f"[{self.phase.upper()}] Load rel cache: {cache_file}")
        else:
            rel_path = os.path.join(self.data_path, f"{self.dataset}.rel")
            df = pd.read_csv(rel_path)

            adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
            src, dst = [], []
            for _, row in df.iterrows():
                o = safe_int(row["origin_id"])
                d = safe_int(row["destination_id"])
                if int(o) in self.geo2ind and int(d) in self.geo2ind:
                    io = self.geo2ind[int(o)]
                    id_ = self.geo2ind[int(d)]
                    adj[io, id_] = 1.0
                    src.append(io)
                    dst.append(id_)

            edge_index = torch.tensor([src, dst], dtype=torch.long) if len(src) else None

            self.adj_mx = adj
            self.edge_index = edge_index

            pickle.dump(
                {"adj_mx": self.adj_mx, "edge_index": self.edge_index},
                open(cache_file, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL
            )
            self.logger.info(f"[{self.phase.upper()}] Save rel cache: {cache_file}")

        self.logger.info(f"[{self.phase.upper()}] Loaded Adj Matrix.")

    # --------------------------
    # DYNA + u_ff_r
    # --------------------------
    def _dyna_cache_file(self):
        return os.path.join(self.cache_dir, f"{self.dataset}_dyna.pkl")

    def _load_or_build_dyna(self):
        import os, pickle, time
        import numpy as np
        import pandas as pd

        # ---- DDP 信息（没有 DDP 也能跑）----
        rank, world = 0, 1
        dist = None
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank, world = dist.get_rank(), dist.get_world_size()
        except Exception:
            dist = None

        cache_file = self._dyna_cache_file()

        def _atomic_dump(obj, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # 原子替换

        def _try_load(path):
            if (not os.path.exists(path)) or os.path.getsize(path) <= 0:
                return None
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError, OSError):
                return None

        def _build_obj_from_raw():
            dyna_path = os.path.join(self.data_path, f"{self.dataset}.dyna")
            df = pd.read_csv(dyna_path)
            df["time"] = pd.to_datetime(df["time"])

            times = sorted(df["time"].unique())
            num_timesteps = len(times)
            begin_timestamp = pd.Timestamp(times[0]).timestamp()

            id_col = "entity_id" if "entity_id" in df.columns else "geo_id"
            df_pivot = df.pivot(index="time", columns=id_col, values="traffic_speed")
            df_pivot = df_pivot.reindex(columns=self.geo_ids, fill_value=0)
            df_pivot = df_pivot.fillna(0.0).replace([np.inf, -np.inf], 0.0)

            traffic_data = np.zeros((num_timesteps, self.num_nodes, 1), dtype=np.float32)
            traffic_data[:, :, 0] = df_pivot.values.astype(np.float32)

            time_features = np.zeros((num_timesteps, 2), dtype=np.float32)
            for i, t in enumerate(pd.DatetimeIndex(times)):
                time_features[i, 0] = (t.hour * 60 + t.minute) / (24 * 60)
                time_features[i, 1] = t.dayofweek / 7.0

            spd = np.nan_to_num(traffic_data[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
            u_ff_r = np.nanpercentile(spd, 95, axis=0).astype(np.float32)
            u_ff_r = np.clip(u_ff_r, 1e-3, None).astype(np.float32)

            return dict(
                times=times,
                num_timesteps=num_timesteps,
                begin_timestamp=begin_timestamp,
                traffic_data=traffic_data,
                time_features=time_features,
                u_ff_r=u_ff_r,
            )

        # ---- 1) rank0 决定是否 rebuild（其他 rank 不要抢读/抢写）----
        obj = None
        if rank == 0:
            obj = _try_load(cache_file)
            if obj is None:
                self.logger.warning(f"[{self.phase.upper()}] Bad/missing dyna cache, rebuild: {cache_file}")
                obj = _build_obj_from_raw()
                _atomic_dump(obj, cache_file)
                self.logger.info(f"[{self.phase.upper()}] Save dyna cache(atomic): {cache_file}")
            else:
                self.logger.info(f"[{self.phase.upper()}] Load dyna cache: {cache_file}")

        # ---- 2) 所有 rank 都 barrier（关键：避免分叉死锁）----
        if world > 1 and dist is not None and dist.is_initialized():
            dist.barrier()

        # ---- 3) 非 rank0：barrier 后再读（带重试）----
        if rank != 0:
            obj = _try_load(cache_file)
            if obj is None:
                for _ in range(10):
                    time.sleep(0.2)
                    obj = _try_load(cache_file)
                    if obj is not None:
                        break
            if obj is None:
                raise RuntimeError(f"[{self.phase.upper()}] dyna cache still broken after barrier: {cache_file}")
            self.logger.info(f"[{self.phase.upper()}] Load dyna cache(after barrier): {cache_file}")

        # ---- 4) 写入成员变量 + 清洗 u_ff_r ----
        self.times = obj["times"]
        self.num_timesteps = obj["num_timesteps"]
        self.begin_timestamp = obj["begin_timestamp"]
        self.traffic_data = obj["traffic_data"]
        self.time_features = obj["time_features"]

        self.u_ff_r = np.asarray(obj.get("u_ff_r", None), dtype=np.float32) if "u_ff_r" in obj else None
        if self.u_ff_r is None:
            spd = np.nan_to_num(self.traffic_data[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
            self.u_ff_r = np.nanpercentile(spd, 95, axis=0).astype(np.float32)

        bad = ~np.isfinite(self.u_ff_r)
        if bad.any():
            rep = np.nanmedian(self.u_ff_r[~bad]) if (~bad).any() else 1.0
            self.u_ff_r[bad] = rep
        self.u_ff_r = np.clip(self.u_ff_r, 1e-3, None).astype(np.float32)

        self.logger.info(f"[{self.phase.upper()}] Loaded Traffic State: {self.traffic_data.shape}")
        self.logger.info(f"[{self.phase.upper()}] u_ff_r(max speed from dyna): {self.u_ff_r.shape}")

    # --------------------------
    # TRAJ DICT (overlap buckets)
    # --------------------------
    def _traj_dict_cache_file(self):
        return os.path.join(self.cache_dir, f"{self.dataset}_traj_dict_{self.phase}.pkl")

    def _load_or_build_traj_dict(self):
        cache_file = self._traj_dict_cache_file()
        if os.path.exists(cache_file):
            obj = pickle.load(open(cache_file, "rb"))
            self.traj_dict = obj["traj_dict"]
            self.min_traj_time_idx = obj["min_t"]
            self.max_traj_time_idx = obj["max_t"]
            self.traj_file_used = obj["traj_file_used"]
            self.logger.info(f"[{self.phase.upper()}] Load traj_dict cache: {cache_file}")
            return

        filename = f"{self.dataset}_traj_{self.phase}.csv"
        traj_path = os.path.join(self.data_path, filename)
        self.traj_file_used = traj_path
        if not os.path.exists(traj_path):
            fallback = os.path.join(self.data_path, f"{self.dataset}_traj_train.csv")
            if os.path.exists(fallback):
                traj_path = fallback
                self.traj_file_used = traj_path

        df = pd.read_csv(self.traj_file_used, sep=";")

        traj_dict = {}
        min_t = float("inf")
        max_t = float("-inf")

        for _, row in df.iterrows():
            try:
                path = ast.literal_eval(row["path"])
                tlist = ast.literal_eval(row["tlist"])
                if len(path) < 2 or len(tlist) < 2 or len(path) != len(tlist):
                    continue

                # 将 timestamp 转成 minute-of-day, day-of-week（归一化到 0~1）
                # tlist 元素是秒级时间戳（你之前就是这么算 start_ts/end_ts 的）
                minutes = []
                weeks = []
                for ts in tlist:
                    dt = pd.to_datetime(int(ts), unit="s")
                    minutes.append((dt.hour * 60 + dt.minute) / (24 * 60))
                    weeks.append(dt.dayofweek / 7.0)

                start_ts = int(tlist[0])
                end_ts = int(tlist[-1])
                start_idx = int((start_ts - self.begin_timestamp) // self.time_intervals)
                end_idx = int((end_ts - self.begin_timestamp) // self.time_intervals)

                mapped = []
                m_min = []
                m_week = []
                for gid, mn, wk in zip(path, minutes, weeks):
                    gid = int(gid)
                    if gid in self.geo2ind:
                        mapped.append(self.geo2ind[gid])
                        m_min.append(float(mn))
                        m_week.append(float(wk))

                if len(mapped) == 0:
                    continue

                min_t = min(min_t, start_idx)
                max_t = max(max_t, end_idx)

                traj_obj = {"loc": mapped, "min": m_min, "week": m_week}

                t = start_idx
                if 0 <= t < self.num_timesteps:
                    traj_dict.setdefault(t, []).append(traj_obj)


            except Exception:
                continue

        self.traj_dict = traj_dict
        self.min_traj_time_idx = min_t
        self.max_traj_time_idx = max_t

        pickle.dump(
            {"traj_dict": self.traj_dict, "min_t": self.min_traj_time_idx,
            "max_t": self.max_traj_time_idx, "traj_file_used": self.traj_file_used},
            open(cache_file, "wb"),
            protocol=pickle.HIGHEST_PROTOCOL
        )
        self.logger.info(f"[{self.phase.upper()}] Save traj_dict cache: {cache_file}")

    # --------------------------
    # TRAJ BATCHES (chunks per time t)
    # --------------------------
    def _traj_batches_cache_file(self):
        return os.path.join(self.cache_dir, f"{self.dataset}_traj_batches_{self.phase}_M{self.traj_batch_size}.pkl")

    def _load_or_build_traj_batches(self):
        cache_file = self._traj_batches_cache_file()
        if os.path.exists(cache_file):
            obj = pickle.load(open(cache_file, "rb"))
            self.traj_batches = obj["traj_batches"]
            self.logger.info(f"[{self.phase.upper()}] Load traj_batches cache: {cache_file}")
        else:
            M = self.traj_batch_size
            traj_batches = {}

            for t in range(self.num_timesteps):
                trajs = self.traj_dict.get(t, [])
                traj_num = len(trajs)

                if traj_num == 0:
                    traj_batches[t] = [[]]
                    continue

                num_traj_batches = (traj_num + M - 1) // M
                batches = []
                for b in range(num_traj_batches):
                    one = trajs[b * M:(b + 1) * M]
                    if len(one) < M:
                        need = M - len(one)
                        for _ in range(need):
                            one.append(trajs[self.rng.randint(0, traj_num - 1)])
                    batches.append(one)
                traj_batches[t] = batches

            self.traj_batches = traj_batches
            pickle.dump({"traj_batches": self.traj_batches}, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"[{self.phase.upper()}] Save traj_batches cache: {cache_file}")

    # --------------------------
    # VALID INDICES (traffic windows)
    # --------------------------
    def _valid_indices_cache_file(self):
        return os.path.join(
            self.cache_dir,
            f"{self.dataset}_valid_idx_{self.phase}_Th{self.input_window}_Tf{self.output_window}.pkl"
        )

    def _load_or_build_valid_indices(self):
        cache_file = self._valid_indices_cache_file()
        if os.path.exists(cache_file):
            obj = pickle.load(open(cache_file, "rb"))
            self.valid_indices = obj["valid_indices"]
            self.logger.info(f"[{self.phase.upper()}] Load valid_indices cache: {cache_file}")
        else:
            self.valid_indices = []
            T_total = self.num_timesteps
            Th, Tf = self.input_window, self.output_window

            if self.min_traj_time_idx == float("inf"):
                pickle.dump({"valid_indices": self.valid_indices}, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
                return

            if self.strict_split_window:
                start_bound = max(0, int(self.min_traj_time_idx))
                end_bound = min(T_total, int(self.max_traj_time_idx) + 1)
                last_start = end_bound - (Th + Tf)
                for s in range(start_bound, last_start + 1):
                    self.valid_indices.append(s)
            else:
                start_bound = max(0, int(self.min_traj_time_idx))
                last_start = T_total - (Th + Tf)
                for s in range(start_bound, last_start + 1):
                    self.valid_indices.append(s)

            pickle.dump({"valid_indices": self.valid_indices}, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"[{self.phase.upper()}] Save valid_indices cache: {cache_file}")

        self.logger.info(f"[{self.phase.upper()}] Generated {len(self.valid_indices)} samples (traffic windows).")

    # --------------------------
    # COVER INDICES (expanded samples)
    # --------------------------
    def _cover_indices_cache_file(self):
        return os.path.join(
            self.cache_dir,
            f"{self.dataset}_cover_idx_{self.phase}_Th{self.input_window}_Tf{self.output_window}_M{self.traj_batch_size}.pkl"
        )

    def _load_or_build_cover_indices(self):
        cache_file = self._cover_indices_cache_file()
        if os.path.exists(cache_file):
            obj = pickle.load(open(cache_file, "rb"))
            self.sample_indices = obj["sample_indices"]
            self.logger.info(f"[{self.phase.upper()}] Load cover_indices cache: {cache_file}")
        else:
            Th, Tf = self.input_window, self.output_window
            sample_indices = []

            for t_start in self.valid_indices:
                t_end = t_start + Th + Tf
                max_k = 1
                for t in range(t_start, t_end):
                    k = len(self.traj_batches.get(t, [[]]))
                    max_k = max(max_k, k)
                for traj_batch_id in range(max_k):
                    sample_indices.append((t_start, traj_batch_id))

            self.sample_indices = sample_indices
            pickle.dump({"sample_indices": self.sample_indices}, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"[{self.phase.upper()}] Save cover_indices cache: {cache_file}")

        self.logger.info(f"[{self.phase.upper()}] Expanded to {len(self.sample_indices)} samples with cover-mode.")

    # --------------------------
    # get traj batch at time t
    # --------------------------
    def _get_traj_batch(self, t_idx: int, traj_batch_id: int, L_override: int = None):
        M = self.traj_batch_size
        L = int(L_override) if L_override is not None else self.max_traj_len
        pad = self.pad_value
        F_t = 3  # [road_id, minutes, weeks]

        batches = self.traj_batches.get(t_idx, [[]])
        if len(batches) == 0 or (len(batches) == 1 and len(batches[0]) == 0):
            out = np.zeros((M, L, F_t), dtype=np.float32)
            out[..., 0] = pad  # road_id pad
            car_mask = np.zeros((M,), dtype=np.bool_)
            return out, car_mask

        sel = batches[traj_batch_id % len(batches)]
        out = np.zeros((M, L, F_t), dtype=np.float32)
        out[..., 0] = pad
        car_mask = np.ones((M,), dtype=np.bool_)

        for i in range(M):
            traj = sel[i]
            if not isinstance(traj, dict):
                raise TypeError(
                    f"traj cache schema mismatch: got {type(traj)}. "
                    f"Please delete traj caches in {self.cache_dir} and rebuild."
                )

            loc = traj.get("loc", [])[:L]
            mn  = traj.get("min", [])[:L]
            wk  = traj.get("week", [])[:L]

            n = min(len(loc), len(mn), len(wk))
            loc, mn, wk = loc[:n], mn[:n], wk[:n]

            out[i, :n, 0] = np.asarray(loc, dtype=np.float32)
            out[i, :n, 1] = np.asarray(mn, dtype=np.float32)
            out[i, :n, 2] = np.asarray(wk, dtype=np.float32)

        return out, car_mask

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t_start, traj_batch_id = self.sample_indices[idx]
        Th, Tf = self.input_window, self.output_window
        t_mid = t_start + Th
        t_end = t_mid + Tf

        TrafficState = torch.from_numpy(self.traffic_data[t_start:t_mid]).float()  # [Th,N,1]
        FutureState  = torch.from_numpy(self.traffic_data[t_mid:t_end]).float()   # [Tf,N,1]
        Et = torch.from_numpy(self.time_features[t_start:t_end]).float()          # [Th+Tf,2]

        # --------- history ---------
        hist_ids, hist_cm = [], []
        for t in range(t_start, t_mid - 1):
            ids, cm = self._get_traj_batch(t, traj_batch_id, L_override=self.max_traj_len)
            hist_ids.append(ids)
            hist_cm.append(cm)

        # --------- last history step (need L + Tf tokens) ---------
        L = self.max_traj_len
        Tf = self.output_window

        ids_full, cm_last = self._get_traj_batch(
            t_mid - 1,
            traj_batch_id,
            L_override=L + Tf
        )  # ids_full: [M, L+Tf, 3]

        # Trajectory at last observed time uses first L tokens
        hist_ids.append(ids_full[:, :L, :])
        hist_cm.append(cm_last)

        # --------- FutureTraj: shift windows from ids_full ---------
        fut_ids, fut_cm = [], []
        for k in range(Tf):
            fut_ids.append(ids_full[:, k+1:k+1+L, :])     # [M, L, 3]
            fut_cm.append(cm_last)                        # same car_mask

        Trajectory = torch.from_numpy(np.asarray(hist_ids, dtype=np.float32))   # [Th,M,L,3]
        FutureTraj = torch.from_numpy(np.asarray(fut_ids, dtype=np.float32))    # [Tf,M,L,3]
        Trajectory_car_mask = torch.from_numpy(np.asarray(hist_cm, dtype=np.bool_))  # [Th,M]
        FutureTraj_car_mask = torch.from_numpy(np.asarray(fut_cm, dtype=np.bool_))   # [Tf,M]

        return {
            "TrafficState": TrafficState,
            "FutureState": FutureState,
            "Trajectory": Trajectory,
            "FutureTraj": FutureTraj,
            "Et": Et,
            "Trajectory_car_mask": Trajectory_car_mask,
            "FutureTraj_car_mask": FutureTraj_car_mask,
            "pad_value": torch.tensor(self.pad_value, dtype=torch.long),
            "t_start": torch.tensor(t_start, dtype=torch.long),
            "traj_batch_id": torch.tensor(traj_batch_id, dtype=torch.long),
        }


# --------------------------
# Collate
# --------------------------
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pad_value = int(batch[0]["pad_value"].item())

    TrafficState = torch.stack([x["TrafficState"] for x in batch], dim=0)
    FutureState  = torch.stack([x["FutureState"] for x in batch], dim=0)
    Trajectory   = torch.stack([x["Trajectory"] for x in batch], dim=0)
    FutureTraj   = torch.stack([x["FutureTraj"] for x in batch], dim=0)
    Et           = torch.stack([x["Et"] for x in batch], dim=0)

    Trajectory_car_mask = torch.stack([x["Trajectory_car_mask"] for x in batch], dim=0)
    FutureTraj_car_mask = torch.stack([x["FutureTraj_car_mask"] for x in batch], dim=0)

    road_id = Trajectory[..., 0].long()   # [B,Th,M,L]
    fut_road = FutureTraj[..., 0].long()

    Trajectory_token_mask = (road_id != pad_value)
    FutureTraj_token_mask = (fut_road != pad_value)

    Trajectory_step_mask = (road_id[..., :-1] != pad_value) & (road_id[..., 1:] != pad_value)
    FutureTraj_step_mask = (fut_road[..., :-1] != pad_value) & (fut_road[..., 1:] != pad_value)

    t_start = torch.stack([x["t_start"] for x in batch], dim=0)
    traj_batch_id = torch.stack([x["traj_batch_id"] for x in batch], dim=0)

    return {
        "TrafficState": TrafficState,
        "FutureState": FutureState,
        "Trajectory": Trajectory,        # [B,Th,M,L,3]
        "FutureTraj": FutureTraj,        # [B,Tf,M,L,3]
        "Et": Et,

        "Trajectory_car_mask": Trajectory_car_mask,
        "FutureTraj_car_mask": FutureTraj_car_mask,
        "Trajectory_token_mask": Trajectory_token_mask,
        "FutureTraj_token_mask": FutureTraj_token_mask,
        "Trajectory_step_mask": Trajectory_step_mask,
        "FutureTraj_step_mask": FutureTraj_step_mask,

        "pad_value": torch.tensor(pad_value, dtype=torch.long),
        "t_start": t_start,
        "traj_batch_id": traj_batch_id,
    }


# --------------------------
# Builder (for train.py)
# --------------------------
def build_dataloaders(config: Dict, distributed: bool = False, rank: int = 0, world_size: int = 1):
    train_ds = STREDataset(config, phase="train")
    eval_ds  = STREDataset(config, phase="eval")
    test_ds  = STREDataset(config, phase="test")

    def _loader(ds, shuffle: bool):
        if len(ds) == 0:
            shuffle = False

        sampler = None
        if distributed:
            sampler = DistributedSampler(
                ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=shuffle,   # train drop_last=True; eval/test False
            )

        return DataLoader(
            ds,
            batch_size=int(config.get("batch_size", 16)),
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=collate_fn,
            pin_memory=bool(config.get("pin_memory", False)),
            drop_last=shuffle,  # train drop_last=True 更稳（每张卡 batch 数一致）
        )

    train_loader = _loader(train_ds, shuffle=True)
    eval_loader  = _loader(eval_ds, shuffle=False)
    test_loader  = _loader(test_ds, shuffle=False)

    static = {
        "adj_mx": torch.tensor(np.asarray(train_ds.adj_mx), dtype=torch.float32),
        "edge_index": train_ds.edge_index,

        "num_nodes": train_ds.num_nodes,
        "pad_value": train_ds.pad_value,

        "node_features": torch.from_numpy(train_ds.node_features).float(),
        "node_fea_dim": train_ds.node_fea_dim,

        "L_r": torch.from_numpy(train_ds.L_r).float(),
        "u_ff_r": torch.from_numpy(train_ds.u_ff_r).float(),
    }
    return train_loader, eval_loader, test_loader, static


