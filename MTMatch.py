import argparse
import csv
import glob
import itertools
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils import data
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from MTMatch_column_name_matching import (
    build_column_profiles,
    build_column_texts,
    column_similarity,
    encode_with_bt,
    normalize_column_name,
)
from selfsl.barlow_twins_simclr import BarlowTwinsSimCLR, lm_mp
from selfsl.block import evaluate_pairs, read_ground_truth, run_blocking
from selfsl.bootstrap import bootstrap
from selfsl.bt_dataset import BTDataset
from selfsl.dataset import DMDataset
from MTMatch_log import init_logger, log, log_args


EXCLUDED_MULTI_TABLE_FILENAMES = {"multi_gt.csv"}
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TABLEMATCH_DIR = str(PACKAGE_ROOT / "data" / "Tablematch")
DEFAULT_REFERENCE_CSV = f"{DEFAULT_TABLEMATCH_DIR}/dataset_1.csv"
DEFAULT_TARGET_CSV = f"{DEFAULT_TABLEMATCH_DIR}/dataset_2-Beijing.csv"
DEFAULT_MULTI_GROUND_TRUTH = f"{DEFAULT_TABLEMATCH_DIR}/multi_gt.csv"


def _top_k_indices(scores: np.ndarray, top_k: int) -> Set[int]:
    if len(scores) == 0:
        return set()
    effective_top_k = max(1, min(top_k, len(scores)))
    ranked_indices = np.argsort(scores)[::-1][:effective_top_k]
    return {int(index) for index in ranked_indices}


def _connected_components_from_graph(graph: Dict[str, Set[str]]) -> List[List[str]]:
    visited: Set[str] = set()
    clusters: List[List[str]] = []

    for node in list(graph.keys()):
        if node in visited:
            continue
        stack = [node]
        component: List[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(component) > 1:
            clusters.append(component)

    return clusters


def discover_semantic_clusters(
    tgt_names: List[str],
    ref_names: List[str],
    similarity_matrix: np.ndarray,
    threshold: float = 0.85,
    top_k_per_node: int = 1,
) -> List[List[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    if len(tgt_names) == 0 or len(ref_names) == 0:
        return []

    tgt_top_neighbors = [
        _top_k_indices(similarity_matrix[tgt_index], top_k_per_node)
        for tgt_index in range(len(tgt_names))
    ]
    ref_top_neighbors = [
        _top_k_indices(similarity_matrix[:, ref_index], top_k_per_node)
        for ref_index in range(len(ref_names))
    ]

    for tgt_index, tgt_name in enumerate(tgt_names):
        for ref_index, ref_name in enumerate(ref_names):
            score = float(similarity_matrix[tgt_index, ref_index])
            if score < threshold:
                continue
            if ref_index not in tgt_top_neighbors[tgt_index] and tgt_index not in ref_top_neighbors[ref_index]:
                continue
            tgt_node = f"Target::{tgt_name}"
            ref_node = f"Ref::{ref_name}"
            graph[tgt_node].add(ref_node)
            graph[ref_node].add(tgt_node)

    return _connected_components_from_graph(graph)


def build_sparse_cross_table_graph(
    node_labels: List[str],
    table_id_of_node: List[int],
    similarity_matrix: Optional[np.ndarray],
    threshold: float,
    top_k_per_node: int = 1,
    score_lookup: Optional[Callable[[int, int], float]] = None,
) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)

    for src_idx, src_node in enumerate(node_labels):
        candidates: List[Tuple[int, float]] = []
        src_table_id = table_id_of_node[src_idx]
        for dst_idx, _ in enumerate(node_labels):
            if src_idx == dst_idx:
                continue
            if table_id_of_node[dst_idx] == src_table_id:
                continue
            if score_lookup is not None:
                score = float(score_lookup(src_idx, dst_idx))
            elif similarity_matrix is not None:
                score = float(similarity_matrix[src_idx, dst_idx])
            else:
                raise ValueError("Either similarity_matrix or score_lookup must be provided.")
            if score >= threshold:
                candidates.append((dst_idx, score))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[1], reverse=True)
        for dst_idx, _ in candidates[: max(1, top_k_per_node)]:
            dst_node = node_labels[dst_idx]
            graph[src_node].add(dst_node)
            graph[dst_node].add(src_node)

    return graph


def _component_merge_is_coherent(
    left_members: Set[int],
    right_members: Set[int],
    score_lookup: Callable[[int, int], float],
    merge_threshold: float,
    average_threshold: float,
) -> bool:
    cross_scores: List[float] = []

    for left_idx in left_members:
        scores_to_right = [float(score_lookup(left_idx, right_idx)) for right_idx in right_members]
        if not scores_to_right or max(scores_to_right) < merge_threshold:
            return False
        cross_scores.extend(scores_to_right)

    for right_idx in right_members:
        scores_to_left = [float(score_lookup(right_idx, left_idx)) for left_idx in left_members]
        if not scores_to_left or max(scores_to_left) < merge_threshold:
            return False

    if not cross_scores:
        return False

    average_score = float(sum(cross_scores) / len(cross_scores))
    return average_score >= average_threshold


def build_conservative_cross_table_clusters(
    node_labels: List[str],
    table_id_of_node: List[int],
    edge_scores: Dict[Tuple[str, str, str, str], float],
    score_lookup: Callable[[int, int], float],
    merge_threshold: float,
    average_threshold: Optional[float] = None,
    enforce_unique_tables: bool = True,
) -> List[List[str]]:
    if average_threshold is None:
        average_threshold = merge_threshold

    label_to_index = {label: idx for idx, label in enumerate(node_labels)}
    candidate_edges: List[Tuple[float, int, int]] = []

    for (src_table, src_col, dst_table, dst_col), score in edge_scores.items():
        if score < merge_threshold:
            continue
        src_idx = label_to_index.get(f"{src_table}::{src_col}")
        dst_idx = label_to_index.get(f"{dst_table}::{dst_col}")
        if src_idx is None or dst_idx is None or src_idx == dst_idx:
            continue
        if table_id_of_node[src_idx] == table_id_of_node[dst_idx]:
            continue
        candidate_edges.append((float(score), src_idx, dst_idx))

    candidate_edges.sort(key=lambda item: item[0], reverse=True)

    parent = list(range(len(node_labels)))
    members: Dict[int, Set[int]] = {idx: {idx} for idx in range(len(node_labels))}
    tables_in_component: Dict[int, Set[int]] = {
        idx: {table_id_of_node[idx]} for idx in range(len(node_labels))
    }

    def find(node_idx: int) -> int:
        while parent[node_idx] != node_idx:
            parent[node_idx] = parent[parent[node_idx]]
            node_idx = parent[node_idx]
        return node_idx

    def union(left_root: int, right_root: int) -> int:
        if len(members[left_root]) < len(members[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        members[left_root].update(members.pop(right_root))
        tables_in_component[left_root].update(tables_in_component.pop(right_root))
        return left_root

    for _, src_idx, dst_idx in candidate_edges:
        src_root = find(src_idx)
        dst_root = find(dst_idx)
        if src_root == dst_root:
            continue
        if enforce_unique_tables and tables_in_component[src_root] & tables_in_component[dst_root]:
            continue
        if not _component_merge_is_coherent(
            left_members=members[src_root],
            right_members=members[dst_root],
            score_lookup=score_lookup,
            merge_threshold=merge_threshold,
            average_threshold=average_threshold,
        ):
            continue
        union(src_root, dst_root)

    clusters: List[List[str]] = []
    for component_members in members.values():
        if len(component_members) <= 1:
            continue
        clusters.append(sorted(node_labels[node_idx] for node_idx in component_members))

    clusters.sort(key=lambda cluster: (-len(cluster), cluster))
    return clusters


def expand_match_scores_with_clusters(
    deduped_match_scores: Dict[Tuple[str, str, str, str], float],
    clusters: List[List[str]],
    node_tables: List[str],
    node_columns: List[str],
    node_labels: List[str],
    score_lookup: Callable[[int, int], float],
    min_edge_score: Optional[float] = None,
) -> Dict[Tuple[str, str, str, str], float]:
    if not clusters:
        return deduped_match_scores

    label_to_index = {label: idx for idx, label in enumerate(node_labels)}
    expanded_scores = dict(deduped_match_scores)

    for cluster in clusters:
        unique_labels = sorted(set(cluster))
        for left_pos in range(len(unique_labels)):
            for right_pos in range(left_pos + 1, len(unique_labels)):
                left_label = unique_labels[left_pos]
                right_label = unique_labels[right_pos]
                left_table = left_label.split("::", 1)[0] if "::" in left_label else ""
                right_table = right_label.split("::", 1)[0] if "::" in right_label else ""
                if left_table and right_table and left_table == right_table:
                    continue

                left_idx = label_to_index.get(left_label)
                right_idx = label_to_index.get(right_label)
                if left_idx is None or right_idx is None:
                    continue

                score = float(score_lookup(left_idx, right_idx))
                if min_edge_score is not None and score < min_edge_score:
                    continue

                edge_key = canonicalize_multi_match_edge(
                    node_tables[left_idx],
                    node_columns[left_idx],
                    node_tables[right_idx],
                    node_columns[right_idx],
                )
                previous_score = expanded_scores.get(edge_key)
                if previous_score is None or score > previous_score:
                    expanded_scores[edge_key] = score

    return expanded_scores


def compute_column_name_similarity(column_name_left: str, column_name_right: str) -> float:
    normalized_left = normalize_column_name(column_name_left).lower()
    normalized_right = normalize_column_name(column_name_right).lower()

    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0

    tokens_left = {
        token for token in re.split(r"[^a-z0-9]+", normalized_left) if token
    }
    tokens_right = {
        token for token in re.split(r"[^a-z0-9]+", normalized_right) if token
    }
    if not tokens_left or not tokens_right:
        return 0.0

    intersection_size = len(tokens_left & tokens_right)
    union_size = len(tokens_left | tokens_right)
    if union_size == 0:
        return 0.0
    return intersection_size / union_size


def infer_column_semantic_family(column_name: str) -> str:
    normalized = normalize_column_name(column_name).lower()
    alnum = re.sub(r"[^a-z0-9]+", "", normalized)

    if not normalized:
        return "unknown"
    if normalized in {"no", "id", "idx", "index"}:
        return "identifier"
    if normalized in {"year"}:
        return "year"
    if normalized in {"month"}:
        return "month"
    if normalized in {"day", "date"}:
        return "date"
    if normalized in {"hour", "time"}:
        return "time"
    if "season" in normalized:
        return "season"
    if "dewp" in normalized or "dewpoint" in alnum:
        return "dew_point"
    if normalized in {"temp", "t"} or "temp" in normalized:
        return "temperature"
    if normalized in {"humi", "rh"} or "humid" in normalized:
        return "humidity"
    if normalized == "ah":
        return "absolute_humidity"
    if "pres" in normalized or "pressure" in normalized:
        return "pressure"
    if "cbwd" in normalized or "winddir" in alnum:
        return "wind_direction"
    if normalized == "iws" or "windspeed" in alnum:
        return "wind_speed"
    if normalized in {"ir", "iprec"} or "precip" in normalized or "rain" in normalized:
        return "precipitation"
    if normalized == "is" or "snow" in normalized:
        return "snow"
    if "pm" in normalized:
        return "particulate_matter"
    if "(gt)" in normalized or "pt08" in normalized:
        return "gas_sensor"
    if any(token in normalized for token in ("co", "nox", "no2", "nmhc", "c6h6", "o3")):
        return "gas_sensor"
    return "unknown"


def compute_semantic_family_compatibility(column_name_left: str, column_name_right: str) -> float:
    left_family = infer_column_semantic_family(column_name_left)
    right_family = infer_column_semantic_family(column_name_right)

    if left_family == "unknown" or right_family == "unknown":
        return 1.0
    if left_family == right_family:
        return 1.0
    return 0.35


def normalize_multi_match_weights(
    embedding_weight: float,
    profile_weight: float,
    column_name_weight: float,
) -> Tuple[float, float, float]:
    weights = [
        max(0.0, float(embedding_weight)),
        max(0.0, float(profile_weight)),
        max(0.0, float(column_name_weight)),
    ]
    total = sum(weights)
    if total <= 0:
        return 1.0, 0.0, 0.0
    return tuple(weight / total for weight in weights)


def build_multi_match_score_lookup(
    node_columns: List[str],
    node_profiles: List[Dict[str, object]],
    embeddings: np.ndarray,
    embedding_weight: float,
    profile_weight: float,
    column_name_weight: float,
    use_semantic_family_prior: bool = True,
) -> Callable[[int, int], float]:
    embedding_weight, profile_weight, column_name_weight = normalize_multi_match_weights(
        embedding_weight=embedding_weight,
        profile_weight=profile_weight,
        column_name_weight=column_name_weight,
    )
    score_cache: Dict[Tuple[int, int], float] = {}

    def lookup(left_idx: int, right_idx: int) -> float:
        key = (left_idx, right_idx) if left_idx <= right_idx else (right_idx, left_idx)
        cached = score_cache.get(key)
        if cached is not None:
            return cached

        embedding_score = float(np.dot(embeddings[left_idx], embeddings[right_idx]))
        profile_score = 0.0
        if profile_weight > 0.0:
            profile_score = float(column_similarity(node_profiles[left_idx], node_profiles[right_idx]))
        column_name_score = 0.0
        if column_name_weight > 0.0:
            column_name_score = compute_column_name_similarity(
                node_columns[left_idx],
                node_columns[right_idx],
            )
        semantic_family_compatibility = 1.0
        if use_semantic_family_prior:
            semantic_family_compatibility = compute_semantic_family_compatibility(
                node_columns[left_idx],
                node_columns[right_idx],
            )

        fused_score = (
            embedding_weight * embedding_score
            + profile_weight * profile_score
            + column_name_weight * column_name_score
        )
        fused_score *= semantic_family_compatibility
        score_cache[key] = fused_score
        return fused_score

    return lookup


def _select_global_cross_table_topk_candidates(
    src_idx: int,
    ann_indices: np.ndarray,
    ann_distances: np.ndarray,
    table_id_of_node: List[int],
    embeddings: np.ndarray,
    top_k: int,
    score_lookup: Optional[Callable[[int, int], float]] = None,
) -> List[Tuple[int, float]]:
    src_table_id = table_id_of_node[src_idx]
    selected: List[Tuple[int, float]] = []

    for dst_idx, distance in zip(ann_indices[src_idx], ann_distances[src_idx]):
        dst_idx = int(dst_idx)
        if dst_idx == src_idx:
            continue
        if table_id_of_node[dst_idx] == src_table_id:
            continue
        if score_lookup is None:
            score = 1.0 - float(distance)
        else:
            score = float(score_lookup(src_idx, dst_idx))
        selected.append((dst_idx, score))
        if len(selected) >= top_k:
            break

    if len(selected) >= top_k:
        return selected

    fallback_candidates: List[Tuple[int, float]] = []
    for dst_idx in range(len(table_id_of_node)):
        if dst_idx == src_idx:
            continue
        if table_id_of_node[dst_idx] == src_table_id:
            continue
        if score_lookup is None:
            score = float(np.dot(embeddings[src_idx], embeddings[dst_idx]))
        else:
            score = float(score_lookup(src_idx, dst_idx))
        fallback_candidates.append((dst_idx, score))

    fallback_candidates.sort(key=lambda item: item[1], reverse=True)
    return fallback_candidates[:top_k]


def _select_per_table_cross_table_topk_candidates(
    src_idx: int,
    ann_indices: np.ndarray,
    ann_distances: np.ndarray,
    table_id_of_node: List[int],
    embeddings: np.ndarray,
    top_k: int,
    score_lookup: Optional[Callable[[int, int], float]] = None,
) -> List[Tuple[int, float]]:
    src_table_id = table_id_of_node[src_idx]
    other_table_ids = sorted({table_id for table_id in table_id_of_node if table_id != src_table_id})
    if not other_table_ids:
        return []

    candidates_by_table: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    for dst_idx, distance in zip(ann_indices[src_idx], ann_distances[src_idx]):
        dst_idx = int(dst_idx)
        if dst_idx == src_idx:
            continue
        dst_table_id = table_id_of_node[dst_idx]
        if dst_table_id == src_table_id:
            continue
        if score_lookup is None:
            score = 1.0 - float(distance)
        else:
            score = float(score_lookup(src_idx, dst_idx))
        candidates_by_table[dst_table_id].append((dst_idx, score))

    need_fallback = any(len(candidates_by_table.get(table_id, [])) < top_k for table_id in other_table_ids)
    if need_fallback:
        fallback_by_table: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for dst_idx in range(len(table_id_of_node)):
            if dst_idx == src_idx:
                continue
            dst_table_id = table_id_of_node[dst_idx]
            if dst_table_id == src_table_id:
                continue
            if score_lookup is None:
                score = float(np.dot(embeddings[src_idx], embeddings[dst_idx]))
            else:
                score = float(score_lookup(src_idx, dst_idx))
            fallback_by_table[dst_table_id].append((dst_idx, score))

        for dst_table_id in other_table_ids:
            existing = candidates_by_table.get(dst_table_id, [])
            if len(existing) >= top_k:
                continue
            existing_ids = {dst_idx for dst_idx, _ in existing}
            fallback_candidates = sorted(
                fallback_by_table.get(dst_table_id, []),
                key=lambda item: item[1],
                reverse=True,
            )
            for candidate in fallback_candidates:
                if candidate[0] in existing_ids:
                    continue
                existing.append(candidate)
                existing_ids.add(candidate[0])
                if len(existing) >= top_k:
                    break
            candidates_by_table[dst_table_id] = existing

    selected: List[Tuple[int, float]] = []
    for dst_table_id in other_table_ids:
        table_candidates = sorted(
            candidates_by_table.get(dst_table_id, []),
            key=lambda item: item[1],
            reverse=True,
        )
        selected.extend(table_candidates[:top_k])

    selected.sort(key=lambda item: item[1], reverse=True)
    return selected


def build_cross_table_topk_matches_with_index(
    node_tables: List[str],
    node_columns: List[str],
    table_id_of_node: List[int],
    embeddings: np.ndarray,
    top_k: int,
    node_profiles: Optional[List[Dict[str, object]]] = None,
    embedding_weight: float = 0.80,
    profile_weight: float = 0.15,
    column_name_weight: float = 0.05,
    use_semantic_family_prior: bool = True,
    initial_search_k: int = 32,
) -> Dict[Tuple[str, str, str, str], float]:
    if len(node_tables) == 0 or len(embeddings) == 0:
        return {}

    num_nodes = len(node_tables)
    effective_top_k = max(1, top_k)
    num_tables = len(set(table_id_of_node))
    expected_cross_table_neighbors = effective_top_k * max(1, num_tables - 1)
    search_k = min(
        num_nodes,
        max(expected_cross_table_neighbors + 1, expected_cross_table_neighbors * 4, initial_search_k),
    )

    index = NearestNeighbors(metric="cosine", algorithm="auto")
    index.fit(embeddings)
    distances, indices = index.kneighbors(embeddings, n_neighbors=search_k)

    if node_profiles is None:
        node_profiles = [{"type": "empty"} for _ in node_columns]
    score_lookup = build_multi_match_score_lookup(
        node_columns=node_columns,
        node_profiles=node_profiles,
        embeddings=embeddings,
        embedding_weight=embedding_weight,
        profile_weight=profile_weight,
        column_name_weight=column_name_weight,
        use_semantic_family_prior=use_semantic_family_prior,
    )

    directed_match_scores: Dict[int, Dict[int, float]] = {}
    for src_idx in range(num_nodes):
        selected = _select_per_table_cross_table_topk_candidates(
            src_idx=src_idx,
            ann_indices=indices,
            ann_distances=distances,
            table_id_of_node=table_id_of_node,
            embeddings=embeddings,
            top_k=effective_top_k,
            score_lookup=score_lookup,
        )
        if not selected:
            selected = _select_global_cross_table_topk_candidates(
                src_idx=src_idx,
                ann_indices=indices,
                ann_distances=distances,
                table_id_of_node=table_id_of_node,
                embeddings=embeddings,
                top_k=effective_top_k,
                score_lookup=score_lookup,
            )
        directed_match_scores[src_idx] = {
            dst_idx: score for dst_idx, score in selected
        }

    deduped_match_scores: Dict[Tuple[str, str, str, str], float] = {}
    for src_idx in range(num_nodes):
        for dst_idx, score in directed_match_scores.get(src_idx, {}).items():
            reverse_score = directed_match_scores.get(dst_idx, {}).get(src_idx)
            if reverse_score is None:
                continue
            edge_key = canonicalize_multi_match_edge(
                node_tables[src_idx],
                node_columns[src_idx],
                node_tables[dst_idx],
                node_columns[dst_idx],
            )
            previous_score = deduped_match_scores.get(edge_key)
            final_score = max(score, reverse_score)
            if previous_score is None or final_score > previous_score:
                deduped_match_scores[edge_key] = final_score

    return deduped_match_scores


class SudowoodoBTMatcher:
    """Match columns of two tables using Sudowoodo's BarlowTwins/SimCLR encoder."""

    def __init__(
        self,
        lm: str = "roberta",
        ckpt_path: str = "",
        batch_size: int = 32,
        max_len: int = 128,
        max_cells_per_column: int = 128,
        lm_only: bool = False,
    ):
        self.lm = lm
        self.ckpt_path = ckpt_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_cells_per_column = max_cells_per_column
        self.lm_only = lm_only

    def _build_texts(
        self,
        csv_path: str,
        max_rows: int,
    ) -> Tuple[List[str], List[str]]:
        names, texts = build_column_texts(
            csv_path=csv_path,
            max_rows=max_rows,
            max_cells=self.max_cells_per_column,
        )
        return names, texts

    def match(
        self,
        reference_csv: str,
        target_csv: str,
        max_rows: int = 50000,
        cluster_threshold: Optional[float] = None,
    ) -> Tuple[List[Tuple[str, str, float]], List[List[str]]]:
        """
        返回:
            matches: 贪心匹配的 (target_column, reference_column, similarity) 列表。
            clusters: 连通分量算法发现的语义类型集群列表。
        """
        ref_names, ref_texts = self._build_texts(reference_csv, max_rows=max_rows)
        tgt_names, tgt_texts = self._build_texts(target_csv, max_rows=max_rows)

        lm_only_flag = self.lm_only or (not self.ckpt_path)

        ref_embeddings = encode_with_bt(
            ref_texts, self.lm, self.ckpt_path, self.batch_size, self.max_len, lm_only_flag
        )
        tgt_embeddings = encode_with_bt(
            tgt_texts, self.lm, self.ckpt_path, self.batch_size, self.max_len, lm_only_flag
        )

        if not len(ref_embeddings) or not len(tgt_embeddings):
            return [], []

        similarity_matrix = np.matmul(tgt_embeddings, ref_embeddings.T)

        matches: List[Tuple[str, str, float]] = []
        for tgt_index, tgt_name in enumerate(tgt_names):
            row = similarity_matrix[tgt_index]
            best_ref_index = int(np.argmax(row))
            best_score = float(row[best_ref_index])
            best_ref_name = ref_names[best_ref_index]
            matches.append((tgt_name, best_ref_name, best_score))

        matches.sort(key=lambda item: item[2], reverse=True)

        clusters: List[List[str]] = []
        if cluster_threshold is not None:
            clusters = discover_semantic_clusters(
                tgt_names=tgt_names,
                ref_names=ref_names,
                similarity_matrix=similarity_matrix,
                threshold=cluster_threshold,
            )

        return matches, clusters

    def match_multiple_tables(
        self,
        table_paths: List[str],
        max_rows: int = 50000,
        cluster_threshold: Optional[float] = None,
        top_k: int = 1,
        embedding_weight: float = 0.80,
        profile_weight: float = 0.15,
        column_name_weight: float = 0.05,
        use_semantic_family_prior: bool = True,
        cluster_strategy: str = "conservative",
        cluster_average_threshold: Optional[float] = None,
        expand_edges_from_clusters: bool = True,
        cluster_edge_threshold: Optional[float] = None,
    ) -> Tuple[List[Tuple[str, str, str, str, float]], List[List[str]]]:
        """
        多表格匹配（跨表）：
            - 统一编码所有表的列
            - 对每个列在其它表中做 Top-K 最近邻匹配
            - 用阈值图 + 连通分量发现 N:M 语义簇

        返回:
            topk_matches: (src_table, src_col, dst_table, dst_col, similarity)
            clusters: 连通分量结果，每个节点格式为 "table::column"
        """
        if len(table_paths) < 2:
            return [], []

        node_tables: List[str] = []
        node_columns: List[str] = []
        node_labels: List[str] = []
        all_texts: List[str] = []
        table_id_of_node: List[int] = []
        node_profiles: List[Dict[str, object]] = []

        for table_id, csv_path in enumerate(table_paths):
            col_names, col_texts = self._build_texts(csv_path, max_rows=max_rows)
            table_profiles = build_column_profiles(csv_path, max_rows=max_rows)
            table_name = Path(csv_path).name
            for col_name, col_text in zip(col_names, col_texts):
                node_tables.append(table_name)
                node_columns.append(col_name)
                node_labels.append(f"{table_name}::{col_name}")
                all_texts.append(col_text)
                table_id_of_node.append(table_id)
                node_profiles.append(table_profiles.get(col_name, {"type": "empty"}))

        if not all_texts:
            return [], []

        lm_only_flag = self.lm_only or (not self.ckpt_path)
        embeddings = encode_with_bt(
            all_texts, self.lm, self.ckpt_path, self.batch_size, self.max_len, lm_only_flag
        )
        if not len(embeddings):
            return [], []

        deduped_match_scores = build_cross_table_topk_matches_with_index(
            node_tables=node_tables,
            node_columns=node_columns,
            table_id_of_node=table_id_of_node,
            embeddings=embeddings,
            top_k=top_k,
            node_profiles=node_profiles,
            embedding_weight=embedding_weight,
            profile_weight=profile_weight,
            column_name_weight=column_name_weight,
            use_semantic_family_prior=use_semantic_family_prior,
        )

        clusters: List[List[str]] = []
        fused_score_lookup: Optional[Callable[[int, int], float]] = None
        if cluster_threshold is not None:
            fused_score_lookup = build_multi_match_score_lookup(
                node_columns=node_columns,
                node_profiles=node_profiles,
                embeddings=embeddings,
                embedding_weight=embedding_weight,
                profile_weight=profile_weight,
                column_name_weight=column_name_weight,
                use_semantic_family_prior=use_semantic_family_prior,
            )
            if cluster_strategy == "connected_components":
                graph = build_sparse_cross_table_graph(
                    node_labels=node_labels,
                    table_id_of_node=table_id_of_node,
                    similarity_matrix=None,
                    threshold=cluster_threshold,
                    score_lookup=fused_score_lookup,
                )
                clusters = _connected_components_from_graph(graph)
            else:
                clusters = build_conservative_cross_table_clusters(
                    node_labels=node_labels,
                    table_id_of_node=table_id_of_node,
                    edge_scores=deduped_match_scores,
                    score_lookup=fused_score_lookup,
                    merge_threshold=cluster_threshold,
                    average_threshold=cluster_average_threshold,
                    enforce_unique_tables=True,
                )
            if expand_edges_from_clusters and clusters:
                deduped_match_scores = expand_match_scores_with_clusters(
                    deduped_match_scores=deduped_match_scores,
                    clusters=clusters,
                    node_tables=node_tables,
                    node_columns=node_columns,
                    node_labels=node_labels,
                    score_lookup=fused_score_lookup,
                    min_edge_score=cluster_edge_threshold,
                )

        topk_matches: List[Tuple[str, str, str, str, float]] = [
            (src_table, src_col, dst_table, dst_col, score)
            for (src_table, src_col, dst_table, dst_col), score in deduped_match_scores.items()
        ]
        topk_matches.sort(key=lambda item: item[4], reverse=True)
        return topk_matches, clusters


def discover_table_paths(multi_tables: str = "", multi_tables_dir: str = "", multi_tables_glob: str = "*.csv") -> List[str]:
    paths: List[str] = []
    if multi_tables:
        paths.extend([item.strip() for item in multi_tables.split(",") if item.strip()])
    if multi_tables_dir:
        pattern = os.path.join(multi_tables_dir, multi_tables_glob)
        paths.extend(glob.glob(pattern))

    unique_paths: List[str] = []
    seen: Set[str] = set()
    for p in paths:
        norm = os.path.abspath(os.path.normpath(p))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(norm) and Path(norm).name not in EXCLUDED_MULTI_TABLE_FILENAMES:
            unique_paths.append(norm)

    unique_paths.sort()
    return unique_paths


def build_pair_prediction_set(
    matches: List[Tuple[str, str, float]],
    reference_table_name: str,
    target_table_name: str,
) -> Set[Tuple[str, str]]:
    pred_pairs: Set[Tuple[str, str]] = set()
    for tgt_col, ref_col, _ in matches:
        left = f"{reference_table_name}::{normalize_column_name(ref_col)}"
        right = f"{target_table_name}::{normalize_column_name(tgt_col)}"
        pair = tuple(sorted((left, right)))
        pred_pairs.add((pair[0], pair[1]))
    return pred_pairs


def normalize_pairwise_clusters(
    clusters: List[List[str]],
    reference_table_name: str,
    target_table_name: str,
) -> List[List[str]]:
    normalized_clusters: List[List[str]] = []
    for cluster in clusters:
        normalized_cluster: List[str] = []
        for node in cluster:
            if node.startswith("Ref::"):
                normalized_cluster.append(
                    f"{reference_table_name}::{normalize_column_name(node[len('Ref::') :])}"
                )
            elif node.startswith("Target::"):
                normalized_cluster.append(
                    f"{target_table_name}::{normalize_column_name(node[len('Target::') :])}"
                )
            else:
                normalized_cluster.append(node)
        normalized_clusters.append(normalized_cluster)
    return normalized_clusters


def filter_ground_truth_for_table_pair(
    gt_pairs: Set[Tuple[str, str]],
    reference_table_name: str,
    target_table_name: str,
) -> Set[Tuple[str, str]]:
    valid_table_names = {reference_table_name, target_table_name}
    filtered_pairs: Set[Tuple[str, str]] = set()
    for left, right in gt_pairs:
        left_table = left.split("::", 1)[0]
        right_table = right.split("::", 1)[0]
        if {left_table, right_table} == valid_table_names:
            filtered_pairs.add((left, right))
    return filtered_pairs


def average_metric_triplets(metric_triplets: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if not metric_triplets:
        return 0.0, 0.0, 0.0
    count = float(len(metric_triplets))
    precision = sum(item[0] for item in metric_triplets) / count
    recall = sum(item[1] for item in metric_triplets) / count
    f1 = sum(item[2] for item in metric_triplets) / count
    return precision, recall, f1


def compute_pair_confusion(
    pred_pairs: Set[Tuple[str, str]],
    gt_pairs: Set[Tuple[str, str]],
) -> Tuple[int, int, int]:
    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    return tp, fp, fn


def compute_metrics_from_confusion(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_pairwise_table_matching(
    matcher: "SudowoodoBTMatcher",
    table_paths: List[str],
    max_rows: int,
    cluster_threshold: Optional[float],
    ground_truth_path: str = "",
) -> None:
    pair_paths = list(itertools.combinations(table_paths, 2))
    all_gt_pairs = load_multi_ground_truth(ground_truth_path) if ground_truth_path else set()
    topk_metric_list: List[Tuple[float, float, float]] = []
    cluster_metric_list: List[Tuple[float, float, float]] = []
    covered_pairs = 0
    skipped_pairs = 0
    topk_tp = topk_fp = topk_fn = 0
    cluster_tp = cluster_fp = cluster_fn = 0

    log(f"Starting pairwise matching for {len(table_paths)} tables ({len(pair_paths)} table pairs).")
    if not ground_truth_path:
        log("No --multi_ground_truth provided. Top-K and Cluster pairwise summaries will be unavailable.")
    elif not all_gt_pairs:
        log(f"No valid ground truth pairs were loaded from {ground_truth_path}. Pairwise summaries will be unavailable.")

    for pair_index, (reference_csv, target_csv) in enumerate(pair_paths, start=1):
        reference_name = Path(reference_csv).name
        target_name = Path(target_csv).name
        log("=" * 80)
        log(f"[Pair {pair_index}/{len(pair_paths)}] {reference_name} <-> {target_name}")

        matches, clusters = matcher.match(
            reference_csv=reference_csv,
            target_csv=target_csv,
            max_rows=max_rows,
            cluster_threshold=cluster_threshold,
        )

        if not matches:
            log("No matches were produced for this table pair.")
            continue

        log("--- Pairwise matches (target -> reference) ---")
        for tgt_name, ref_name, score in matches:
            log(f"{target_name}::{tgt_name}  -->  {reference_name}::{ref_name}  (similarity={score:.4f})")

        if clusters:
            log(f"--- Cluster connected components (threshold={cluster_threshold}) ---")
            log(f"Discovered {len(clusters)} clusters")
            for idx, cluster in enumerate(clusters, start=1):
                log(f"[Cluster {idx}] - {len(cluster)} columns")
                for col in cluster:
                    log(f"  - {col}")

        if all_gt_pairs:
            pair_gt = filter_ground_truth_for_table_pair(all_gt_pairs, reference_name, target_name)
            if not pair_gt:
                skipped_pairs += 1
                log(f"Skipped evaluation for {reference_name} <-> {target_name}: no ground truth pairs found.")
                continue

            covered_pairs += 1
            pair_predictions = build_pair_prediction_set(matches, reference_name, target_name)
            topk_metrics = evaluate_pair_predictions(
                pair_predictions,
                pair_gt,
                f"Pairwise Top-K edge metrics: {reference_name} <-> {target_name}",
            )
            topk_metric_list.append(topk_metrics)
            cur_tp, cur_fp, cur_fn = compute_pair_confusion(pair_predictions, pair_gt)
            topk_tp += cur_tp
            topk_fp += cur_fp
            topk_fn += cur_fn
            log(
                f"Top-K pairwise summary: P={topk_metrics[0]:.4f}, "
                f"R={topk_metrics[1]:.4f}, F1={topk_metrics[2]:.4f}"
            )

            normalized_clusters = normalize_pairwise_clusters(clusters, reference_name, target_name)
            cluster_predictions = cluster_pairs_from_clusters(normalized_clusters)
            cluster_metrics = evaluate_pair_predictions(
                cluster_predictions,
                pair_gt,
                f"Pairwise cluster metrics: {reference_name} <-> {target_name}",
            )
            cluster_metric_list.append(cluster_metrics)
            cur_tp, cur_fp, cur_fn = compute_pair_confusion(cluster_predictions, pair_gt)
            cluster_tp += cur_tp
            cluster_fp += cur_fp
            cluster_fn += cur_fn
            log(
                f"Cluster pairwise summary: P={cluster_metrics[0]:.4f}, "
                f"R={cluster_metrics[1]:.4f}, F1={cluster_metrics[2]:.4f}"
            )
        else:
            log("Top-K pairwise summary: unavailable (missing valid --multi_ground_truth)")
            log("Cluster pairwise summary: unavailable (missing valid --multi_ground_truth)")

    log("=" * 80)
    if all_gt_pairs:
        log(
            f"Pairwise evaluation coverage: gt-covered table pairs = "
            f"{covered_pairs}/{len(pair_paths)}, skipped = {skipped_pairs}"
        )
        if topk_metric_list:
            avg_topk = average_metric_triplets(topk_metric_list)
            log(
                f"Macro-average Top-K pairwise summary over {len(topk_metric_list)} table pairs: "
                f"P={avg_topk[0]:.4f}, R={avg_topk[1]:.4f}, F1={avg_topk[2]:.4f}"
            )
            micro_topk = compute_metrics_from_confusion(topk_tp, topk_fp, topk_fn)
            log(
                f"Micro-average Top-K pairwise summary over {len(topk_metric_list)} table pairs: "
                f"P={micro_topk[0]:.4f}, R={micro_topk[1]:.4f}, F1={micro_topk[2]:.4f} "
                f"(TP={topk_tp}, FP={topk_fp}, FN={topk_fn})"
            )
        else:
            log("Average Top-K pairwise summary: unavailable (no table pairs with ground truth)")

        if cluster_metric_list:
            avg_cluster = average_metric_triplets(cluster_metric_list)
            log(
                f"Macro-average Cluster pairwise summary over {len(cluster_metric_list)} table pairs: "
                f"P={avg_cluster[0]:.4f}, R={avg_cluster[1]:.4f}, F1={avg_cluster[2]:.4f}"
            )
            micro_cluster = compute_metrics_from_confusion(cluster_tp, cluster_fp, cluster_fn)
            log(
                f"Micro-average Cluster pairwise summary over {len(cluster_metric_list)} table pairs: "
                f"P={micro_cluster[0]:.4f}, R={micro_cluster[1]:.4f}, F1={micro_cluster[2]:.4f} "
                f"(TP={cluster_tp}, FP={cluster_fp}, FN={cluster_fn})"
            )
            if micro_cluster[0] < 0.1:
                log("Warning: cluster precision is still very low. Consider increasing --cluster_threshold.")
        else:
            log("Average Cluster pairwise summary: unavailable (no table pairs with ground truth)")
    else:
        log("Average Top-K pairwise summary: unavailable (missing valid --multi_ground_truth)")
        log("Average Cluster pairwise summary: unavailable (missing valid --multi_ground_truth)")

    log("Pairwise matching completed.")

def canonicalize_multi_match_edge(
    src_table: str,
    src_col: str,
    dst_table: str,
    dst_col: str,
) -> Tuple[str, str, str, str]:
    left = (src_table, src_col)
    right = (dst_table, dst_col)
    if left <= right:
        return src_table, src_col, dst_table, dst_col
    return dst_table, dst_col, src_table, src_col


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fin:
        return [line.strip() for line in fin if line.strip()]


def load_positive_pairs(task_path: str) -> Set[Tuple[str, str]]:
    positives = set()
    for split in ["train.txt", "valid.txt", "test.txt"]:
        split_path = os.path.join(task_path, split)
        if not os.path.exists(split_path):
            continue
        with open(split_path, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                left, right, label = parts[0], parts[1], parts[2]
                if label == "1":
                    positives.add((left, right))
    return positives


def encode_texts(
    texts: List[str],
    model: BarlowTwinsSimCLR,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_len: int,
    device: str,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    all_embeddings: List[np.ndarray] = []
    use_amp = torch.cuda.is_available()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model.bert(input_ids)[0][:, 0, :]
            all_embeddings.append(outputs.float().cpu().numpy())

    mat = np.concatenate(all_embeddings, axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def add_cluster_negatives(
    trainset: DMDataset,
    task_path: str,
    model: BarlowTwinsSimCLR,
    hp: argparse.Namespace,
) -> Tuple[DMDataset, Dict[str, int]]:
    left_path = os.path.join(task_path, "tableA.txt")
    right_path = os.path.join(task_path, "tableB.txt")
    left_texts = read_lines(left_path)
    right_texts = read_lines(right_path)
    if not left_texts or not right_texts:
        return trainset, {"added_negatives": 0}

    tokenizer = AutoTokenizer.from_pretrained(lm_mp[hp.lm])
    device = model.device
    all_texts = left_texts + right_texts
    embeddings = encode_texts(
        all_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=hp.batch_size,
        max_len=hp.max_len,
        device=device,
    )

    n_clusters = min(hp.num_clusters, max(2, len(all_texts) // 2))
    kmeans = KMeans(n_clusters=n_clusters, random_state=hp.run_id)
    cluster_ids = kmeans.fit_predict(embeddings)
    left_clusters = cluster_ids[: len(left_texts)]
    right_clusters = cluster_ids[len(left_texts) :]

    right_by_cluster: Dict[int, List[str]] = defaultdict(list)
    for right_text, cluster_id in zip(right_texts, right_clusters):
        right_by_cluster[int(cluster_id)].append(right_text)

    positives = load_positive_pairs(task_path)
    pair_set = set(trainset.pairs)
    new_pairs = list(trainset.pairs)
    new_labels = list(trainset.labels)

    added = 0
    for left_text, cluster_id in zip(left_texts, left_clusters):
        candidates = right_by_cluster.get(int(cluster_id), [])
        if not candidates:
            candidates = right_texts
        for _ in range(hp.cluster_negatives_per_left):
            for _ in range(5):
                right_text = random.choice(candidates)
                if (left_text, right_text) in positives:
                    continue
                if (left_text, right_text) in pair_set:
                    continue
                new_pairs.append((left_text, right_text))
                new_labels.append(0)
                pair_set.add((left_text, right_text))
                added += 1
                break

    trainset.pairs = new_pairs
    trainset.labels = new_labels
    return trainset, {"added_negatives": added}


def evaluate_model(
    model: BarlowTwinsSimCLR,
    iterator: data.DataLoader,
    threshold: Optional[float] = None,
    use_amp: bool = False,
) -> Tuple[float, float, float, Optional[float]]:
    all_probs: List[float] = []
    all_labels: List[int] = []
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            if len(batch) == 4:
                x1, x2, x12, y = batch
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(2, x1, x2, x12)
            else:
                x, y = batch
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(2, x, None, None)

            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.float().cpu().numpy().tolist()
            all_labels += y.cpu().numpy().tolist()

    if threshold is not None:
        preds = [1 if p > threshold else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds, zero_division=0)
        p = precision_score(all_labels, preds, zero_division=0)
        r = recall_score(all_labels, preds, zero_division=0)
        return f1, p, r, None

    best_th = 0.5
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0
    for th in np.arange(0.0, 1.0, 0.05):
        preds = [1 if p > th else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
            best_p = precision_score(all_labels, preds, zero_division=0)
            best_r = recall_score(all_labels, preds, zero_division=0)

    return best_f1, best_p, best_r, best_th


def ssl_train_epoch(
    iterator: data.DataLoader,
    model: BarlowTwinsSimCLR,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    hp: argparse.Namespace,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for y1, y2 in iterator:
        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            if hp.ssl_method == "simclr":
                loss = model(0, y1, y2, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
            elif hp.ssl_method == "barlow_twins":
                loss = model(1, y1, y2, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
            else:
                alpha = 1 - hp.alpha_bt
                loss = alpha * model(0, y1, y2, [], da=hp.da) + (1 - alpha) * model(1, y1, y2, [])

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(1, steps)


def finetune_epoch(
    iterator: data.DataLoader,
    model: BarlowTwinsSimCLR,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    hp: argparse.Namespace,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss()
    for batch in iterator:
        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            if len(batch) == 4:
                x1, x2, x12, y = batch
                prediction = model(2, x1, x2, x12)
            else:
                x, y = batch
                prediction = model(2, x, None, None)
            loss = criterion(prediction, y.to(model.device))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(1, steps)


def run_blocking_pipeline(
    model: BarlowTwinsSimCLR,
    task_path: str,
    hp: argparse.Namespace,
) -> Dict[str, object]:
    left_path = os.path.join(task_path, "tableA.txt")
    right_path = os.path.join(task_path, "tableB.txt")
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        return {"error": "Missing tableA.txt/tableB.txt for blocking."}

    left_dataset = BTDataset(left_path, lm=hp.lm, size=None, max_len=hp.max_len)
    right_dataset = BTDataset(right_path, lm=hp.lm, size=None, max_len=hp.max_len)
    pairs = run_blocking(left_dataset, right_dataset, model, hp)

    output_dir = os.path.join(hp.logdir, hp.task)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "blocking_result.pkl")
    import pickle

    pickle.dump(pairs, open(output_path, "wb"))

    train_csv = os.path.join(task_path, "train.csv")
    if not os.path.exists(train_csv):
        return {"pairs_path": output_path, "pairs_count": len(pairs)}

    ground_truth, _ = read_ground_truth(task_path)
    if hp.k:
        recalls: List[float] = []
        sizes: List[int] = []
        for k in range(1, hp.k + 1):
            recall, size = evaluate_pairs(pairs, ground_truth, k=k)
            recalls.append(float(recall))
            sizes.append(int(size))
        return {
            "pairs_path": output_path,
            "pairs_count": len(pairs),
            "recalls": recalls,
            "sizes": sizes,
        }

    recall = evaluate_pairs(pairs, ground_truth)
    return {
        "pairs_path": output_path,
        "pairs_count": len(pairs),
        "recall": float(recall),
    }


def run_ssl_pipeline(args: argparse.Namespace) -> None:
    set_global_seed(args.run_id)

    if args.bootstrap or args.zero:
        if args.size is None:
            args.size = 500
            print("Bootstrap requested with size=None; defaulting --size to 500.")

    task_path = os.path.join("data", args.task_type, args.task)
    train_path = os.path.join(task_path, "train.txt")
    valid_path = os.path.join(task_path, "valid.txt")
    test_path = os.path.join(task_path, "test.txt")
    train_nolabel_path = os.path.join(task_path, "train_no_label.txt")

    if not os.path.exists(train_nolabel_path):
        log(f"Missing train_no_label.txt at {train_nolabel_path}")
        return

    trainset = DMDataset(train_path, lm=args.lm, size=args.size, max_len=args.max_len, da=None)
    validset = DMDataset(valid_path, lm=args.lm, size=args.size, max_len=args.max_len)
    testset = DMDataset(test_path, lm=args.lm, size=None, max_len=args.max_len)

    trainset_nolabel = BTDataset(
        train_nolabel_path,
        lm=args.lm,
        size=args.unlabeled_size,
        max_len=args.max_len,
        da=args.da,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BarlowTwinsSimCLR(args, device=device, lm=args.lm).to(device)

    ckpt_path = args.ckpt_path
    if not ckpt_path:
        ckpt_path = os.path.join(args.logdir, args.task, "ssl.pt")

    if args.use_saved_ckpt and os.path.exists(ckpt_path):
        saved_state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        state_dict = saved_state["model"]
        model.load_state_dict(state_dict)
        log(f"Loaded checkpoint from {ckpt_path}")

    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_amp)

    ssl_steps = 0
    if args.n_ssl_epochs > 0:
        ssl_steps = len(trainset_nolabel) // max(1, args.batch_size) * args.n_ssl_epochs
    finetune_steps = len(trainset) // max(1, args.batch_size // 2) * max(0, args.n_epochs - args.n_ssl_epochs)
    total_steps = ssl_steps + finetune_steps
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(1, total_steps))

    ssl_loader = data.DataLoader(
        dataset=trainset_nolabel,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainset_nolabel.pad,
    )
    ft_batch_size = max(1, args.batch_size // 2)

    if args.n_ssl_epochs > 0:
        for epoch in range(1, args.n_ssl_epochs + 1):
            ssl_loss = ssl_train_epoch(
                ssl_loader, model, optimizer, scheduler, scaler, args, use_amp
            )
            log(f"SSL epoch {epoch}/{args.n_ssl_epochs}: loss={ssl_loss:.4f}")

    bootstrap_metrics = None
    if args.bootstrap or args.zero:
        trainset, tpr, tnr, fpr, fnr = bootstrap(model, args, blocked=True)
        bootstrap_metrics = {
            "tpr": tpr,
            "tnr": tnr,
            "fpr": fpr,
            "fnr": fnr,
            "size": len(trainset),
        }

    cluster_metrics = None
    if args.clustering:
        trainset, cluster_metrics = add_cluster_negatives(trainset, task_path, model, args)

    train_iter = data.DataLoader(
        dataset=trainset,
        batch_size=ft_batch_size,
        shuffle=True,
        collate_fn=trainset.pad,
    )
    valid_iter = data.DataLoader(
        dataset=validset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=validset.pad,
    )
    test_iter = data.DataLoader(
        dataset=testset,
        batch_size=max(1, args.batch_size * 16),
        shuffle=False,
        collate_fn=testset.pad,
    )

    best_metrics: Dict[str, float] = {"dev_f1": 0.0}
    for epoch in range(args.n_ssl_epochs + 1, args.n_epochs + 1):
        ft_loss = finetune_epoch(
            train_iter, model, optimizer, scheduler, scaler, args, use_amp
        )
        dev_f1, dev_p, dev_r, best_th = evaluate_model(
            model, valid_iter, threshold=None, use_amp=use_amp
        )
        test_f1, test_p, test_r, _ = evaluate_model(
            model, test_iter, threshold=best_th, use_amp=use_amp
        )
        log(
            "Epoch {}/{}: loss={:.4f}, dev_f1={:.4f}, test_f1={:.4f}".format(
                epoch, args.n_epochs, ft_loss, dev_f1, test_f1
            )
        )
        if dev_f1 > best_metrics["dev_f1"]:
            best_metrics = {
                "dev_f1": dev_f1,
                "dev_p": dev_p,
                "dev_r": dev_r,
                "threshold": best_th,
                "test_f1": test_f1,
                "test_p": test_p,
                "test_r": test_r,
            }

    if args.save_ckpt:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"model": model.state_dict()}, ckpt_path)
        log(f"Saved checkpoint to {ckpt_path}")

    log("Pipeline metrics:")
    if best_metrics["dev_f1"] == 0.0 and args.n_epochs <= args.n_ssl_epochs:
        log("  No fine-tuning epochs were run; metrics are unavailable.")
    else:
        log(
            "  Dev  F1/P/R: {:.4f} / {:.4f} / {:.4f} (threshold={})".format(
                best_metrics["dev_f1"],
                best_metrics["dev_p"],
                best_metrics["dev_r"],
                best_metrics["threshold"],
            )
        )
        log(
            "  Test F1/P/R: {:.4f} / {:.4f} / {:.4f}".format(
                best_metrics["test_f1"],
                best_metrics["test_p"],
                best_metrics["test_r"],
            )
        )

    if bootstrap_metrics:
        log(
            "  Bootstrap TPR/TNR/FPR/FNR: {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
                bootstrap_metrics["tpr"],
                bootstrap_metrics["tnr"],
                bootstrap_metrics["fpr"],
                bootstrap_metrics["fnr"],
            )
        )
        log(f"  Bootstrap dataset size: {bootstrap_metrics['size']}")

    if cluster_metrics:
        log(f"  Cluster negatives added: {cluster_metrics['added_negatives']}")

    if args.blocking:
        blocking_metrics = run_blocking_pipeline(model, task_path, args)
        if "error" in blocking_metrics:
            log(f"  Blocking skipped: {blocking_metrics['error']}")
        else:
            log(f"  Blocking pairs saved: {blocking_metrics['pairs_path']}")
            if "recalls" in blocking_metrics:
                recalls = blocking_metrics["recalls"]
                sizes = blocking_metrics["sizes"]
                for k, (recall, size) in enumerate(zip(recalls, sizes), start=1):
                    log(f"  Blocking recall@{k}: {recall:.4f} (candidates={size})")
            elif "recall" in blocking_metrics:
                log(f"  Blocking recall: {blocking_metrics['recall']:.4f}")
            else:
                log(f"  Blocking candidates: {blocking_metrics['pairs_count']}")

def evaluate_metrics(predictions_list, ground_truth_list):
    preds = set(predictions_list)
    gts = set(ground_truth_list)

    tp = len(preds.intersection(gts))
    fp = len(preds - gts)
    fn = len(gts - preds)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    log("=" * 40)
    log("Matching Evaluation Metrics")
    log("=" * 40)
    log(f"True Positives  (TP): {tp}")
    log(f"False Positives (FP): {fp}")
    log(f"False Negatives (FN): {fn}")
    log("-" * 40)
    log(f"Precision:          {precision:.4f}")
    log(f"Recall:             {recall:.4f}")
    log(f"F1-Score:           {f1:.4f}")
    log("=" * 40)

    return precision, recall, f1

def evaluate_clustering(clusters, true_labels):
    total_items = 0
    correct_majority = 0
    pred_pairs = set()

    for cluster in clusters:
        cols = list(cluster)
        total_items += len(cols)

        label_counts = {}
        for col in cols:
            clean_col = col.split("::")[-1] if "::" in col else col
            label = true_labels.get(clean_col, "Unknown_" + clean_col)
            label_counts[label] = label_counts.get(label, 0) + 1

        if label_counts:
            correct_majority += max(label_counts.values())

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1 = cols[i].split("::")[-1] if "::" in cols[i] else cols[i]
                c2 = cols[j].split("::")[-1] if "::" in cols[j] else cols[j]
                pair = tuple(sorted([c1, c2]))
                pred_pairs.add(pair)

    purity = correct_majority / total_items if total_items > 0 else 0.0

    true_clusters_map = {}
    for col, label in true_labels.items():
        if label not in true_clusters_map:
            true_clusters_map[label] = []
        true_clusters_map[label].append(col)

    true_pairs = set()
    for label, cols in true_clusters_map.items():
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pair = tuple(sorted([cols[i], cols[j]]))
                true_pairs.add(pair)

    tp = len(pred_pairs.intersection(true_pairs))
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    pair_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pair_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pair_f1 = 2 * pair_p * pair_r / (pair_p + pair_r) if (pair_p + pair_r) > 0 else 0.0

    log("=" * 45)
    log("Cluster Evaluation Metrics (N:M semantic discovery)")
    log("=" * 45)
    log(f"Purity:             {purity:.4f}")
    log("-" * 45)
    log(f"Pairwise TP: {tp} | FP: {fp} | FN: {fn}")
    log(f"Pairwise Precision: {pair_p:.4f}")
    log(f"Pairwise Recall:    {pair_r:.4f}")
    log(f"Pairwise F1:        {pair_f1:.4f}")
    log("=" * 45)

    return purity, pair_p, pair_r, pair_f1

def load_multi_ground_truth(path: str) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    if not path or not os.path.exists(path):
        return pairs

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"src_table", "src_col", "dst_table", "dst_col"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("multi_ground_truth CSV must contain: src_table,src_col,dst_table,dst_col")
        for row in reader:
            src_table = (row.get("src_table") or "").strip()
            src_col = normalize_column_name(row.get("src_col") or "")
            dst_table = (row.get("dst_table") or "").strip()
            dst_col = normalize_column_name(row.get("dst_col") or "")
            if not src_table or not src_col or not dst_table or not dst_col:
                continue
            left = f"{src_table}::{src_col}"
            right = f"{dst_table}::{dst_col}"
            pair = tuple(sorted((left, right)))
            pairs.add((pair[0], pair[1]))
    return pairs


def cluster_pairs_from_clusters(clusters: List[List[str]]) -> Set[Tuple[str, str]]:
    pred_pairs: Set[Tuple[str, str]] = set()
    for cluster in clusters:
        cluster_nodes = sorted(set(cluster))
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                left_table = cluster_nodes[i].split("::", 1)[0] if "::" in cluster_nodes[i] else ""
                right_table = cluster_nodes[j].split("::", 1)[0] if "::" in cluster_nodes[j] else ""
                if left_table and right_table and left_table == right_table:
                    continue
                pred_pairs.add((cluster_nodes[i], cluster_nodes[j]))
    return pred_pairs


def evaluate_pair_predictions(
    pred_pairs: Set[Tuple[str, str]],
    gt_pairs: Set[Tuple[str, str]],
    label: str,
) -> Tuple[float, float, float]:
    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    log("=" * 45)
    log(f"{label}")
    log("-" * 45)
    log(f"GT pairs: {len(gt_pairs)}, Pred pairs: {len(pred_pairs)}")
    log(f"Pairwise TP={tp}, FP={fp}, FN={fn}")
    log(f"Pairwise Precision={precision:.4f}")
    log(f"Pairwise Recall={recall:.4f}")
    log(f"Pairwise F1={f1:.4f}")
    log("=" * 45)
    return precision, recall, f1


def evaluate_multi_match_metrics(
    multi_matches: List[Tuple[str, str, str, str, float]],
    ground_truth_path: str,
) -> Optional[Tuple[float, float, float]]:
    gt_pairs = load_multi_ground_truth(ground_truth_path)
    if not gt_pairs:
        return None

    pred_pairs: Set[Tuple[str, str]] = set()
    for src_table, src_col, dst_table, dst_col, _ in multi_matches:
        left = f"{src_table}::{src_col}"
        right = f"{dst_table}::{dst_col}"
        pair = tuple(sorted((left, right)))
        pred_pairs.add((pair[0], pair[1]))

    return evaluate_pair_predictions(pred_pairs, gt_pairs, "Multi-table Top-K edge metrics")


def evaluate_multi_cluster_metrics(
    clusters: List[List[str]],
    ground_truth_path: str,
) -> Optional[Tuple[float, float, float]]:
    gt_pairs = load_multi_ground_truth(ground_truth_path)
    if not gt_pairs:
        return None

    pred_pairs = cluster_pairs_from_clusters(clusters)
    return evaluate_pair_predictions(pred_pairs, gt_pairs, "Multi-table cluster metrics")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match table columns using Sudowoodo's BarlowTwins/SimCLR encoder (Plan B).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="match",
        choices=["match", "pairwise_match", "multi_match", "pipeline"],
        help="Run two-table matching, directory-wise pairwise matching, multi-table matching, or the full Sudowoodo SSL pipeline.",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE_CSV,
        help="Path to reference CSV file (canonical column names).",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_CSV,
        help="Path to target CSV file whose columns will be matched.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=50000,
        help="Maximum number of rows from each table to use when building texts.",
    )
    parser.add_argument(
        "--cluster_threshold",
        type=float,
        default=None,
        help="Optional similarity threshold for connected-components clustering. If omitted, clustering is skipped.",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="roberta",
        help="Backbone LM name used by Sudowoodo (e.g., roberta, bert, distilbert).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Optional path to ssl.pt checkpoint. If empty, LM-only encoding is used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding texts.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--max_cells_per_column",
        type=int,
        default=128,
        help="Maximum number of non-empty cells per column when building texts.",
    )
    parser.add_argument(
        "--lm_only",
        action="store_true",
        help="If set, ignore ckpt_path and use LM-only encoder.",
    )
    parser.add_argument(
        "--multi_tables",
        type=str,
        default="",
        help="Comma-separated CSV paths for multi-table matching.",
    )
    parser.add_argument(
        "--multi_tables_dir",
        type=str,
        default=DEFAULT_TABLEMATCH_DIR,
        help="Directory that contains multiple CSV files for multi-table matching.",
    )
    parser.add_argument(
        "--multi_tables_glob",
        type=str,
        default="*.csv",
        help="Glob pattern under --multi_tables_dir (e.g., table_*.csv).",
    )
    parser.add_argument(
        "--multi_top_k",
        type=int,
        default=1,
        help="Top-K nearest matches retained against each other table for every column in multi-table mode.",
    )
    parser.add_argument(
        "--multi_embedding_weight",
        type=float,
        default=0.80,
        help="Weight of embedding similarity in multi-table fused scoring.",
    )
    parser.add_argument(
        "--multi_profile_weight",
        type=float,
        default=0.15,
        help="Weight of value-profile similarity in multi-table fused scoring.",
    )
    parser.add_argument(
        "--multi_name_weight",
        type=float,
        default=0.05,
        help="Weight of column-name similarity in multi-table fused scoring.",
    )
    parser.add_argument(
        "--multi_semantic_family_prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the benchmark-specific semantic-family compatibility prior in multi-table fused scoring.",
    )
    parser.add_argument(
        "--multi_cluster_strategy",
        type=str,
        default="conservative",
        choices=["conservative", "connected_components"],
        help="Cluster construction strategy for multi-table mode.",
    )
    parser.add_argument(
        "--multi_cluster_average_threshold",
        type=float,
        default=None,
        help="Optional average-score threshold used by conservative multi-table clustering. Defaults to --cluster_threshold.",
    )
    parser.add_argument(
        "--multi_expand_edges_from_clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expand multi-table edge output with pairwise links implied by the final clusters.",
    )
    parser.add_argument(
        "--multi_cluster_edge_threshold",
        type=float,
        default=None,
        help="Optional minimum fused score when adding cluster-supported edges back into multi-table edge output.",
    )
    parser.add_argument(
        "--multi_ground_truth",
        type=str,
        default=DEFAULT_MULTI_GROUND_TRUTH,
        help=(
            "Optional CSV path for multi-table evaluation. "
            "Required columns: src_table,src_col,dst_table,dst_col"
        ),
    )
    parser.add_argument(
        "--pairwise_tables_dir",
        type=str,
        default=DEFAULT_TABLEMATCH_DIR,
        help="Directory that contains multiple CSV files for pairwise two-table matching.",
    )
    parser.add_argument(
        "--pairwise_tables_glob",
        type=str,
        default="*.csv",
        help="Glob pattern under --pairwise_tables_dir (e.g., dataset_*.csv).",
    )

    parser.add_argument("--task_type", type=str, default="em", help="Task type for the pipeline.")
    parser.add_argument("--task", type=str, default="Abt-Buy", help="Task name under data/<task_type>.")
    parser.add_argument("--logdir", type=str, default="result_em", help="Output directory for checkpoints and logs.")
    parser.add_argument("--ssl_method", type=str, default="simclr", choices=["simclr", "barlow_twins", "combined"])
    parser.add_argument("--n_ssl_epochs", type=int, default=3, help="Number of SSL pre-training epochs.")
    parser.add_argument("--n_epochs", type=int, default=20, help="Total number of epochs (SSL + fine-tuning).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--projector", type=str, default="768", help="Projector MLP definition.")
    parser.add_argument("--scale_loss", type=float, default=1.0 / 256.0, help="Scale factor for BT loss.")
    parser.add_argument("--lambd", type=float, default=3.9e-3, help="Off-diagonal loss weight for BT.")
    parser.add_argument("--alpha_bt", type=float, default=0.001, help="Weight for BT in combined loss.")
    parser.add_argument("--da", type=str, default="cutoff", help="Data augmentation operator for SSL.")
    parser.add_argument("--cutoff_ratio", type=float, default=0.05, help="Cutoff ratio used by DA.")
    parser.add_argument("--size", type=int, default=None, help="Number of labeled examples for fine-tuning.")
    parser.add_argument("--unlabeled_size", type=int, default=10000, help="Number of unlabeled examples for SSL.")
    parser.add_argument("--bootstrap", action="store_true", help="Enable bootstrap pseudo-labeling.")
    parser.add_argument("--zero", action="store_true", help="Enable zero-shot bootstrap setting.")
    parser.add_argument("--multiplier", type=int, default=8, help="Multiplier for bootstrap pseudo labels.")
    parser.add_argument("--clustering", action="store_true", help="Enable cluster-based negative sampling.")
    parser.add_argument("--num_clusters", type=int, default=90, help="Number of clusters for negative sampling.")
    parser.add_argument(
        "--cluster_negatives_per_left",
        type=int,
        default=1,
        help="Number of cluster-based negatives per left entity.",
    )
    parser.add_argument("--blocking", action="store_true", help="Run blocking candidate generation and evaluation.")
    parser.add_argument("--k", type=int, default=10, help="Top-k for blocking.")
    parser.add_argument("--threshold", type=float, default=None, help="Similarity threshold for blocking.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision if CUDA is available.")
    parser.add_argument("--save_ckpt", action="store_true", help="Save SSL checkpoint after training.")
    parser.add_argument("--use_saved_ckpt", action="store_true", help="Load SSL checkpoint before training.")
    parser.add_argument("--run_id", type=int, default=0, help="Random seed for the pipeline.")

    args = parser.parse_args()

    if args.mode in {"multi_match", "pairwise_match"}:
        args.reference = "<unused in pairwise/multi_match>"
        args.target = "<unused in pairwise/multi_match>"
    elif args.mode == "match":
        args.multi_tables = "<unused in match>"
        args.multi_tables_dir = "<unused in match>"
        args.multi_tables_glob = "<unused in match>"
        args.multi_ground_truth = "<unused in match>"
        args.pairwise_tables_dir = "<unused in match>"
        args.pairwise_tables_glob = "<unused in match>"

    init_logger(f"MTMatch_{args.mode}")
    log_args(args)
    set_global_seed(args.run_id)

    if args.mode == "pipeline":
        run_ssl_pipeline(args)
        return

    matcher = SudowoodoBTMatcher(
        lm=args.lm,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_cells_per_column=args.max_cells_per_column,
        lm_only=args.lm_only,
    )

    if args.mode == "pairwise_match":
        pairwise_dir = args.pairwise_tables_dir or args.multi_tables_dir
        pairwise_glob = args.pairwise_tables_glob or args.multi_tables_glob
        if not pairwise_dir:
            log("两两匹配模式只需要提供文件夹，请使用 --pairwise_tables_dir 指定 CSV 目录。")
            return

        table_paths = discover_table_paths(
            multi_tables="",
            multi_tables_dir=pairwise_dir,
            multi_tables_glob=pairwise_glob,
        )
        if len(table_paths) < 2:
            log("两两匹配至少需要两张 CSV。请检查 --pairwise_tables_dir 下是否有足够的表文件。")
            return

        run_pairwise_table_matching(
            matcher=matcher,
            table_paths=table_paths,
            max_rows=args.max_rows,
            cluster_threshold=args.cluster_threshold,
            ground_truth_path=args.multi_ground_truth,
        )
        return

    if args.mode == "multi_match":
        table_paths = discover_table_paths(
            multi_tables=args.multi_tables,
            multi_tables_dir=args.multi_tables_dir,
            multi_tables_glob=args.multi_tables_glob,
        )
        if len(table_paths) < 2:
            log("多表匹配需要至少两张 CSV。请通过 --multi_tables 或 --multi_tables_dir 提供输入。")
            return

        log(f"发现 {len(table_paths)} 张表，开始进行统一编码与跨表匹配...")
        log(f"Discovered {len(table_paths)} tables. Starting shared-embedding multi-table matching.")
        log(
            "Top-K candidate generation uses per-table nearest-neighbor retrieval "
            "with reciprocal filtering inspired by MultiEM-style merging."
        )
        normalized_weights = normalize_multi_match_weights(
            embedding_weight=args.multi_embedding_weight,
            profile_weight=args.multi_profile_weight,
            column_name_weight=args.multi_name_weight,
        )
        log(
            "Fused multi-table scoring weights: "
            f"embedding={normalized_weights[0]:.2f}, "
            f"profile={normalized_weights[1]:.2f}, "
            f"column_name={normalized_weights[2]:.2f}"
        )
        log(f"Semantic-family prior enabled: {args.multi_semantic_family_prior}")
        effective_cluster_average_threshold = args.multi_cluster_average_threshold
        if effective_cluster_average_threshold is None:
            effective_cluster_average_threshold = args.cluster_threshold
        log(
            "Multi-table cluster strategy: "
            f"{args.multi_cluster_strategy} "
            f"(threshold={args.cluster_threshold}, "
            f"average_threshold={effective_cluster_average_threshold})"
        )
        log(
            "Cluster-backed edge expansion: "
            f"enabled={args.multi_expand_edges_from_clusters}, "
            f"edge_threshold={args.multi_cluster_edge_threshold}"
        )
        if args.cluster_threshold is None:
            log("Clustering is disabled, so the full all-pairs similarity matrix will be skipped.")
        multi_matches, clusters = matcher.match_multiple_tables(
            table_paths=table_paths,
            max_rows=args.max_rows,
            cluster_threshold=args.cluster_threshold,
            top_k=args.multi_top_k,
            embedding_weight=args.multi_embedding_weight,
            profile_weight=args.multi_profile_weight,
            column_name_weight=args.multi_name_weight,
            use_semantic_family_prior=args.multi_semantic_family_prior,
            cluster_strategy=args.multi_cluster_strategy,
            cluster_average_threshold=args.multi_cluster_average_threshold,
            expand_edges_from_clusters=args.multi_expand_edges_from_clusters,
            cluster_edge_threshold=args.multi_cluster_edge_threshold,
        )

        log("--- 多表 Top-K 跨表匹配结果 ---")
        for src_table, src_col, dst_table, dst_col, score in multi_matches:
            log(
                f"{src_table}::{src_col}  -->  {dst_table}::{dst_col} "
                f"(similarity={score:.4f})"
            )

        if clusters:
            log(
                f"--- 多表聚类结果 (strategy={args.multi_cluster_strategy}, "
                f"阈值: {args.cluster_threshold}) ---"
            )
            log(f"共发现 {len(clusters)} 个语义类型集群:")
            for idx, cluster in enumerate(clusters, 1):
                log(f"[Cluster {idx}] - 包含 {len(cluster)} 列:")
                for col in cluster:
                    log(f"  - {col}")

        if args.multi_ground_truth:
            try:
                edge_metrics = evaluate_multi_match_metrics(multi_matches, args.multi_ground_truth)
                cluster_metrics = None
                if args.cluster_threshold is not None:
                    cluster_metrics = evaluate_multi_cluster_metrics(clusters, args.multi_ground_truth)
                if edge_metrics is None and cluster_metrics is None:
                    log(f"未读取到有效 ground truth: {args.multi_ground_truth}")
                else:
                    if edge_metrics is not None:
                        log(
                            "Top-K pairwise summary: "
                            f"P={edge_metrics[0]:.4f}, R={edge_metrics[1]:.4f}, F1={edge_metrics[2]:.4f}"
                        )
                    if cluster_metrics is not None:
                        log(
                            "Cluster pairwise summary: "
                            f"P={cluster_metrics[0]:.4f}, R={cluster_metrics[1]:.4f}, F1={cluster_metrics[2]:.4f}"
                        )
            except Exception as e:
                log(f"多表评估失败: {e}")
        return

    matches, clusters = matcher.match(
        reference_csv=args.reference,
        target_csv=args.target,
        max_rows=args.max_rows,
        cluster_threshold=args.cluster_threshold,
    )

    log("--- 贪心匹配结果 (1:1 映射) ---")
    predicted_pairs = []
    for tgt_name, ref_name, score in matches:
        log(f"{tgt_name}  -->  {ref_name}  (similarity={score:.4f})")
        predicted_pairs.append((tgt_name, ref_name))

    if clusters:
        log(f"--- 连通分量聚类结果 (N:M 映射, 阈值: {args.cluster_threshold}) ---")
        log(f"共发现 {len(clusters)} 个语义类型集群:")
        for idx, cluster in enumerate(clusters, 1):
            log(f"[Cluster {idx}] - 包含 {len(cluster)} 列:")
            for col in cluster:
                log(f"  - {col}")

    ground_truth = [
        ("year", "year"),
        ("month", "month"),
        ("day", "day"),
        ("hour", "hour"),
        ("TEMP", "TEMP"),
        ("PRES", "PRES"),
        ("DEWP", "DEWP"),
        ("Iws", "Iws"),
        ("cbwd", "cbwd"),
        ("No", "No"),
        ("PM_US Post", "pm2.5"),
        ("PM_Dongsi", "pm2.5"),
        ("PM_Dongsihuan", "pm2.5"),
        ("PM_Nongzhanguan", "pm2.5")
    ]

    evaluate_metrics(predicted_pairs, ground_truth)

    if clusters:
        # 定义真实语义标签 (Ground Truth Concepts)
        # 相同 value 的列，在物理世界上应该被分在同一个簇里
        true_semantic_labels = {
            "year": "CONCEPT_YEAR",
            "month": "CONCEPT_MONTH",
            "day": "CONCEPT_DAY",
            "hour": "CONCEPT_HOUR",
            "season": "CONCEPT_SEASON",
            "pm2.5": "CONCEPT_PM25",
            "PM_US Post": "CONCEPT_PM25",
            "PM_Dongsi": "CONCEPT_PM25",
            "PM_Dongsihuan": "CONCEPT_PM25",
            "PM_Nongzhanguan": "CONCEPT_PM25",
            "TEMP": "CONCEPT_TEMP",
            "PRES": "CONCEPT_PRES",
            "DEWP": "CONCEPT_DEWP",
            "HUMI": "CONCEPT_HUMI",
            "Iws": "CONCEPT_WIND_SPEED",
            "cbwd": "CONCEPT_WIND_DIR",
            "precipitation": "CONCEPT_PRECIPITATION",
            "Iprec": "CONCEPT_PRECIPITATION",
            "Ir": "CONCEPT_PRECIPITATION",
            "Is": "CONCEPT_SNOW",
            "No": "CONCEPT_ID"
        }
        evaluate_clustering(clusters, true_semantic_labels)
if __name__ == "__main__":
    main()
