"""Custom collate function for variable-length audio and variable bird count."""

import torch


def v33da_collate_fn(batch: list[dict]) -> dict:
    """
    Collate samples with variable audio length and bird count.

    Pads audio to max length, birds to max count, returns masks.
    """
    max_birds = max(s["n_birds"] for s in batch)
    B = len(batch)

    has_audio = "audio" in batch[0]
    if has_audio:
        max_audio_len = max(s["audio"].shape[1] for s in batch)
        audio = torch.zeros(B, batch[0]["audio"].shape[0], max_audio_len)
        audio_mask = torch.zeros(B, max_audio_len, dtype=torch.bool)
        for i, s in enumerate(batch):
            L = s["audio"].shape[1]
            audio[i, :, :L] = s["audio"]
            audio_mask[i, :L] = True

    has_accelerometer = "accelerometer" in batch[0]
    if has_accelerometer:
        max_accel_len = max(s["accelerometer"].shape[1] for s in batch)
        accelerometer = torch.zeros(B, max_birds, max_accel_len)
        accel_mask = torch.zeros(B, max_accel_len, dtype=torch.bool)

    # Pad bird tensors
    bird_positions = torch.zeros(B, max_birds, 3)
    bird_head_orient = torch.zeros(B, max_birds, 3)
    n_radio_features = batch[0]["bird_radio"].shape[-1]
    bird_radio = torch.zeros(B, max_birds, n_radio_features)

    has_radio_temporal = "bird_radio_temporal" in batch[0]
    if has_radio_temporal:
        n_radio_ch = batch[0]["bird_radio_temporal"].shape[1]
        max_radio_frames = max(s["bird_radio_temporal"].shape[2] for s in batch)
        bird_radio_temporal = torch.zeros(B, max_birds, n_radio_ch, max_radio_frames)

    has_pose_temporal = "bird_pose_temporal" in batch[0]
    if has_pose_temporal:
        pose_dim = batch[0]["bird_pose_temporal"].shape[-1]
        max_pose_frames = max(s["bird_pose_temporal"].shape[1] for s in batch)
        bird_pose_temporal = torch.zeros(B, max_birds, max_pose_frames, pose_dim)
        pose_frame_mask = torch.zeros(B, max_pose_frames, dtype=torch.bool)

    has_video_candidate = "bird_video_crops" in batch[0]
    if has_video_candidate:
        _, _, n_views, n_channels, crop_h, crop_w = batch[0]["bird_video_crops"].shape
        max_video_frames = max(s["bird_video_crops"].shape[1] for s in batch)
        bird_video_crops = torch.zeros(B, max_birds, max_video_frames, n_views, n_channels, crop_h, crop_w)
        video_frame_mask = torch.zeros(B, max_video_frames, dtype=torch.bool)

    has_whisper_emb = "whisper_emb" in batch[0]
    if has_whisper_emb:
        whisper_shape = tuple(batch[0]["whisper_emb"].shape)
        whisper_emb = torch.zeros((B, *whisper_shape), dtype=batch[0]["whisper_emb"].dtype)

    bird_mask = torch.zeros(B, max_birds, dtype=torch.bool)
    label_idx = torch.zeros(B, dtype=torch.long)
    label_position = torch.zeros(B, 3)
    min_dist = torch.zeros(B)

    for i, s in enumerate(batch):
        n = s["n_birds"]
        bird_positions[i, :n] = s["bird_positions"]
        bird_head_orient[i, :n] = s["bird_head_orient"]
        bird_radio[i, :n] = s["bird_radio"]
        if has_accelerometer:
            acc = s["accelerometer"]
            accelerometer[i, :n, :acc.shape[1]] = acc
            accel_mask[i, :acc.shape[1]] = True
        if has_radio_temporal:
            rt = s["bird_radio_temporal"]
            bird_radio_temporal[i, :n, :, :rt.shape[2]] = rt
        if has_pose_temporal:
            pt = s["bird_pose_temporal"]
            bird_pose_temporal[i, :n, :pt.shape[1]] = pt
            pose_frame_mask[i, :pt.shape[1]] = True
        if has_video_candidate:
            vc = s["bird_video_crops"]
            bird_video_crops[i, :n, :vc.shape[1]] = vc
            video_frame_mask[i, :vc.shape[1]] = s["video_frame_mask"]
        bird_mask[i, :n] = s["bird_mask"]
        label_idx[i] = s["label_idx"]
        label_position[i] = s["label_position"]
        min_dist[i] = s["min_inter_bird_dist_mm"]
        if has_whisper_emb:
            whisper_emb[i] = s["whisper_emb"]

    # Non-tensor metadata
    bird_names = [s["bird_names"] for s in batch]
    experiments = [s["experiment"] for s in batch]
    sample_ids = [s["sample_id"] for s in batch]

    out = {
        "bird_positions": bird_positions,
        "bird_head_orient": bird_head_orient,
        "bird_radio": bird_radio,
        "bird_mask": bird_mask,
        "label_idx": label_idx,
        "label_position": label_position,
        "min_inter_bird_dist_mm": min_dist,
        "bird_names": bird_names,
        "experiments": experiments,
        "sample_ids": sample_ids,
    }
    if has_audio:
        out["audio"] = audio
        out["audio_mask"] = audio_mask
    if has_accelerometer:
        out["accelerometer"] = accelerometer
        out["accel_mask"] = accel_mask
    if has_radio_temporal:
        out["bird_radio_temporal"] = bird_radio_temporal
    if has_pose_temporal:
        out["bird_pose_temporal"] = bird_pose_temporal
        out["pose_frame_mask"] = pose_frame_mask
    if has_video_candidate:
        out["bird_video_crops"] = bird_video_crops
        out["video_frame_mask"] = video_frame_mask
    if has_whisper_emb:
        out["whisper_emb"] = whisper_emb
    return out
