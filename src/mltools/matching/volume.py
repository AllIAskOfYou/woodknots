import numpy as np
import itertools

def morton_encode_array(coords: np.ndarray) -> np.ndarray:
    """coords: (N, 3) int64 array of (i, j, k). Returns (N,) int64 Morton codes."""
    x = coords + 2**20  # Shift to positive range for 21 bits
    x = x.astype(np.uint64) & np.uint64(0x1fffff)

    def spread(x):
        x = (x | (x << np.uint64(32))) & np.uint64(0x1f00000000ffff)
        x = (x | (x << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
        x = (x | (x << np.uint64(8))) & np.uint64(0x100f00f00f00f00f)
        x = (x | (x << np.uint64(4))) & np.uint64(0x10c30c30c30c30c3)
        x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
        return x

    xi, yj, zk = spread(x[:, 0]), spread(x[:, 1]), spread(x[:, 2])
    return (xi | (yj << np.uint64(1)) | (zk << np.uint64(2))).astype(np.int64)

def morton_decode_array(codes: np.ndarray) -> np.ndarray:
    """codes: (N,) int64 Morton codes. Returns (N, 3) int64 array of (i, j, k)."""
    codes = codes.astype(np.uint64)

    def compact(x: np.ndarray) -> np.ndarray:
        # Inverse of the "spread" used in morton_encode_array: undoes each
        # shift step in reverse order, pulling every-3rd-bit back together
        # into a contiguous 21-bit value.
        x = x & np.uint64(0x1249249249249249)
        x = (x | (x >> np.uint64(2)))  & np.uint64(0x10c30c30c30c30c3)
        x = (x | (x >> np.uint64(4)))  & np.uint64(0x100f00f00f00f00f)
        x = (x | (x >> np.uint64(8)))  & np.uint64(0x1f0000ff0000ff)
        x = (x | (x >> np.uint64(16))) & np.uint64(0x1f00000000ffff)
        x = (x | (x >> np.uint64(32))) & np.uint64(0x1fffff)
        return x

    i = compact(codes)
    j = compact(codes >> np.uint64(1))
    k = compact(codes >> np.uint64(2))

    coords = np.stack([i, j, k], axis=-1).astype(np.int64)
    coords = coords - 2**20  # Shift back to original range
    return coords

class VoxelGrid:
    def __init__(self, voxel_size: float):
        self.voxel_size = voxel_size
        self.volume_ids: list[int] = []
        self.group_ids: list[int] = []
        self.volumes: list[set[int]] = []

    def snap_to_grid(self, points: np.ndarray) -> np.ndarray:
        """points: (N, 3) float32 array of (x, y, z). Returns (N, 3) int64 array of snapped (i, j, k)."""
        snapped = np.round(points / self.voxel_size).astype(np.int64)
        # There need to be at least 3 points in same voxel to consider it valid
        snapped, counts = np.unique(snapped, axis=0, return_counts=True)
        max_count = np.max(counts)
        snapped = snapped[counts > max_count / 4]
        return snapped

    def add_volume(self, points: np.ndarray, volume_id: int = -1, group_id: int = -1):
        """points: (N, 3) float32 array of (x, y, z)."""
        snapped = self.snap_to_grid(points)
        morton_codes = morton_encode_array(snapped)
        self.volumes.append(set(morton_codes))
        self.volume_ids.append(volume_id)
        self.group_ids.append(group_id)

    def find_structures(self, merge_threshold: float = 0.5, map_indexes: bool = True) -> list[list[int]]:
        """Find connected structures based on overlapping volumes, using
        union-find so multi-way (transitive) bridging is handled correctly."""
        num_volumes = len(self.volumes)
        parent = list(range(num_volumes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # --- build candidate pairs via inverted index (voxel -> volume ids) ---
        inverted_index: dict[int, list[int]] = {}
        for i, vol in enumerate(self.volumes):
            for voxel in vol:
                inverted_index.setdefault(voxel, []).append(i)

        candidate_pairs: set[tuple[int, int]] = set()
        for occupants in inverted_index.values():
            if len(occupants) < 2:
                continue
            for a, b in itertools.combinations(occupants, 2):
                ga, gb = self.group_ids[a], self.group_ids[b]
                if ga > -1 and ga == gb:
                    continue  # same group => guaranteed disjoint, never a candidate
                candidate_pairs.add((a, b) if a < b else (b, a))

        # --- evaluate candidates, merging those that clear the threshold ---
        for i, j in candidate_pairs:
            if find(i) == find(j):
                continue  # already in the same structure, skip redundant work

            intersection_size = len(self.volumes[i] & self.volumes[j])
            min_size = min(len(self.volumes[i]), len(self.volumes[j]))
            if min_size > 0 and intersection_size / min_size > merge_threshold:
                union(i, j)

        # --- collect final groups ---
        groups: dict[int, list[int]] = {}
        for i in range(num_volumes):
            root = find(i)
            groups.setdefault(root, []).append(i)

        if map_indexes:
            # remap to volume_ids
            for root, members in groups.items():
                groups[root] = [self.volume_ids[i] for i in members]

        return list(groups.values())

    def visualize_structures(self, structures: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        """Return a tuple of (points, indexes) for visualization. Points are the snapped voxel centers, and indexes are the structure ids."""
        points_list: list[np.ndarray] = []
        indexes_list: list[np.ndarray] = []
        for structure_id, volume_ids in enumerate(structures):
            for volume_id in volume_ids:
                idx = self.volume_ids.index(volume_id)
                morton_volume = self.volumes[idx]
                points = morton_decode_array(np.array(list(morton_volume), dtype=np.int64))
                snapped_points = points * self.voxel_size
                points_list.append(snapped_points)
                indexes_list.append(np.full(len(snapped_points), structure_id, dtype=np.int64))
        return np.vstack(points_list), np.concatenate(indexes_list)

    def visualize_volume(self, volume_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a tuple of (points, indexes) for visualization. Points are the snapped voxel centers, and indexes are the volume ids."""
        if volume_index < 0 or volume_index >= len(self.volumes):
            raise ValueError(f"Invalid volume_index {volume_index}. Must be in range [0, {len(self.volumes) - 1}].")
        morton_volume = self.volumes[volume_index]
        points = morton_decode_array(np.array(list(morton_volume), dtype=np.int64))
        snapped_points = points * self.voxel_size
        indexes = np.full(len(snapped_points), volume_index, dtype=np.int64)
        return snapped_points, indexes

    def visualize_volumes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a tuple of (points, indexes) for visualization. Points are the snapped voxel centers, and indexes are the volume ids."""
        points_list: list[np.ndarray] = []
        indexes_list: list[np.ndarray] = []
        for volume_index in range(len(self.volumes)):
            pl, il = self.visualize_volume(volume_index)
            points_list.append(pl)
            indexes_list.append(il)
        return np.vstack(points_list), np.concatenate(indexes_list)