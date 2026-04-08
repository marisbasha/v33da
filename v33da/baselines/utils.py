"""Shared utilities for classical baselines: triangulation, nearest-bird lookup."""

import numpy as np


SPEED_OF_SOUND_MM_S = 343000.0  # 343 m/s in mm/s


def nearest_bird(estimated_pos: np.ndarray, bird_positions: np.ndarray) -> tuple[int, float]:
    """
    Find nearest bird to an estimated 3D position.

    Args:
        estimated_pos: (3,) estimated source position (mm)
        bird_positions: (N, 3) bird 3D positions (mm)

    Returns:
        (bird_idx, distance_mm)
    """
    dists = np.linalg.norm(bird_positions - estimated_pos, axis=1)
    idx = np.argmin(dists)
    return int(idx), float(dists[idx])


def triangulate_tdoa(tdoas: np.ndarray, mic_pairs: list[tuple[int, int]],
                     mic_positions: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D position from TDOA measurements using least-squares.

    Args:
        tdoas: (M,) TDOA values in seconds for M mic pairs
        mic_pairs: list of (i, j) mic index pairs
        mic_positions: (N_mics, 3) mic positions in mm

    Returns:
        (3,) estimated 3D position in mm
    """
    # Convert TDOA to distance differences
    dd = tdoas * SPEED_OF_SOUND_MM_S  # distance differences in mm

    # Set up linear system: for each pair (i, j), we have
    # |x - m_i| - |x - m_j| = d_{ij}
    # Linearize around centroid using far-field approximation
    # This is a simplified approach; for better accuracy use iterative methods

    n_pairs = len(mic_pairs)
    if n_pairs < 3:
        return mic_positions.mean(axis=0)  # fallback

    # Reference mic is first mic of first pair
    ref = mic_pairs[0][0]
    m_ref = mic_positions[ref]

    # Build overdetermined linear system A @ x = b
    A = []
    b = []
    for k, (i, j) in enumerate(mic_pairs):
        mi = mic_positions[i]
        mj = mic_positions[j]
        # Linearized: 2*(mj - mi) @ x = |mj|^2 - |mi|^2 - dd[k]^2 + ...
        # Simplified spherical intersection
        A.append(2 * (mj - mi))
        b.append(np.dot(mj, mj) - np.dot(mi, mi) - dd[k] ** 2)

    A = np.array(A)
    b = np.array(b)

    # Least-squares solve
    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        x = mic_positions.mean(axis=0)

    return x


def get_mic_pairs(n_mics: int) -> list[tuple[int, int]]:
    """Generate all unique mic pairs."""
    pairs = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            pairs.append((i, j))
    return pairs
