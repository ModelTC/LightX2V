import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image


def video_to_image_frames(input_video_path, save_directory=None, fps=1):
    """
    Extracts image frames from a video file at the specified frame rate and saves them as JPEG format.
    Supports regular video files, webcam captures, WebM files, and GIF files, including incomplete files.

    Args:
        input_video_path: Path to the input video file
        save_directory: Directory to save extracted frames (default: None)
        fps: Number of frames to extract per second (default: 1)

    Returns: List of file paths to extracted frames
    """
    extracted_frame_paths = []
    frame_indices = []  # Track frame indices for metadata
    source_fps = None

    # For GIF files, use PIL library for better handling
    if input_video_path.lower().endswith(".gif"):
        try:
            print(f"Processing GIF file using PIL: {input_video_path}")

            with Image.open(input_video_path) as gif_img:
                # Get GIF properties
                frame_duration_ms = gif_img.info.get("duration", 100)
                gif_frame_rate = 1000.0 / frame_duration_ms if frame_duration_ms > 0 else 10.0
                source_fps = gif_frame_rate

                print(f"GIF properties: {gif_img.n_frames} frames, {gif_frame_rate:.2f} FPS, {frame_duration_ms}ms per frame")

                sampling_interval = max(1, int(gif_frame_rate / fps)) if fps < gif_frame_rate else 1

                saved_count = 0
                for current_frame_index in range(gif_img.n_frames):
                    gif_img.seek(current_frame_index)

                    if current_frame_index % sampling_interval == 0:
                        rgb_frame = gif_img.convert("RGB")
                        frame_ndarray = np.array(rgb_frame)
                        frame_output_path = os.path.join(save_directory, f"frame_{saved_count:06d}.jpg")
                        pil_image = Image.fromarray(frame_ndarray)
                        pil_image.save(frame_output_path, "JPEG", quality=95)
                        extracted_frame_paths.append(frame_output_path)
                        frame_indices.append(current_frame_index)
                        saved_count += 1

                if extracted_frame_paths:
                    print(f"Successfully extracted {len(extracted_frame_paths)} frames from GIF using PIL")
                    # Save metadata
                    _save_old_metadata(save_directory, frame_indices, source_fps)
                    return extracted_frame_paths

        except Exception as error:
            print(f"PIL GIF extraction error: {str(error)}, falling back to OpenCV")

    # For WebM files, use FFmpeg directly for more stable processing
    if input_video_path.lower().endswith(".webm"):
        try:
            print(f"Processing WebM file using FFmpeg: {input_video_path}")

            # Get video FPS first
            cap = cv2.VideoCapture(input_video_path)
            source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            output_frame_pattern = os.path.join(save_directory, "frame_%04d.jpg")

            ffmpeg_command = ["ffmpeg", "-i", input_video_path, "-vf", f"fps={fps}", "-q:v", "2", output_frame_pattern]

            ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process_stdout, process_stderr = ffmpeg_process.communicate()

            # Collect all extracted frames and calculate indices
            extracted_frame_paths = []
            for filename in sorted(os.listdir(save_directory)):
                if filename.startswith("frame_") and filename.endswith(".jpg"):
                    full_frame_path = os.path.join(save_directory, filename)
                    extracted_frame_paths.append(full_frame_path)
                    # Extract frame number from filename (frame_XXXX.jpg)
                    try:
                        frame_num = int(filename.split("_")[1].split(".")[0])
                        # Estimate original frame index based on fps ratio
                        frame_idx = int(frame_num * source_fps / fps)
                        frame_indices.append(frame_idx)
                    except Exception:
                        frame_indices.append(len(frame_indices))

            if extracted_frame_paths:
                print(f"Successfully extracted {len(extracted_frame_paths)} frames from WebM using FFmpeg")
                _save_old_metadata(save_directory, frame_indices, source_fps)
                return extracted_frame_paths

            print("FFmpeg extraction failed, falling back to OpenCV")
        except Exception as error:
            print(f"FFmpeg extraction error: {str(error)}, falling back to OpenCV")

    # Standard OpenCV method for non-WebM files or as fallback
    try:
        video_capture = cv2.VideoCapture(input_video_path)

        if input_video_path.lower().endswith(".webm"):
            video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"VP80"))

        source_fps = video_capture.get(cv2.CAP_PROP_FPS) or fps
        extraction_interval = max(1, int(source_fps / fps))
        processed_frame_count = 0

        cv2.setLogLevel(0)

        while True:
            read_success, current_frame = video_capture.read()
            if not read_success:
                break

            if processed_frame_count % extraction_interval == 0:
                try:
                    if current_frame is not None and current_frame.size > 0:
                        rgb_converted_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                        frame_output_path = os.path.join(save_directory, f"frame_{len(extracted_frame_paths):06d}.jpg")
                        cv2.imwrite(frame_output_path, cv2.cvtColor(rgb_converted_frame, cv2.COLOR_RGB2BGR))
                        extracted_frame_paths.append(frame_output_path)
                        frame_indices.append(processed_frame_count)
                except Exception as error:
                    print(f"Warning: Failed to process frame {processed_frame_count}: {str(error)}")

            processed_frame_count += 1

            if processed_frame_count > 1000:
                break

        video_capture.release()
        print(f"Extracted {len(extracted_frame_paths)} frames from video using OpenCV")

        # Save metadata
        if extracted_frame_paths:
            _save_old_metadata(save_directory, frame_indices, source_fps)

    except Exception as error:
        print(f"Error extracting frames: {str(error)}")

    return extracted_frame_paths


def _save_old_metadata(save_directory, frame_indices, fps):
    """Save metadata for old sampling strategy."""
    if not frame_indices or not fps:
        return

    try:
        meta = {"frame_indices": frame_indices, "frame_times": [idx / fps for idx in frame_indices], "fps": fps, "algorithm": "uniform_fps_based"}
        metadata_path = os.path.join(save_directory, "frame_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save metadata: {e}")


def _resize_for_flow(frame, long_edge=320):
    height, width = frame.shape[:2]
    long_side = max(height, width)
    if long_side <= long_edge:
        return frame
    scale = long_edge / float(long_side)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _resize_for_clarity(frame, long_edge=480):
    """Resize frame for clarity calculation (480p for better accuracy)."""
    height, width = frame.shape[:2]
    long_side = max(height, width)
    if long_side <= long_edge:
        return frame
    scale = long_edge / float(long_side)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _create_dis_flow():
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "createOptFlow_DIS"):
        return cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    if hasattr(cv2, "DISOpticalFlow_create"):
        return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    return None


def _calculate_histogram(image):
    """
    Calculate normalized color histogram for global deduplication.
    Using HSV for better robustness to brightness changes.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 8 bins for H, 4 for S, 4 for V -> 128 dim vector
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _calculate_hist_similarity(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def _advance_cap_to_frame(cap, current_pos, target_idx):
    """Advance cap so that next read() returns frame target_idx. Returns target_idx."""
    dist = target_idx - current_pos
    if dist <= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        return target_idx
    if dist < 50:
        for _ in range(dist):
            cap.grab()
        return target_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    return target_idx


def _merge_search_windows(candidate_indices, window_size=3):
    """
    Merge adjacent search windows to reduce disk seeks.
    Returns list of (start_idx, end_idx, target_indices) tuples.
    """
    if not candidate_indices:
        return []

    merged = []
    sorted_indices = sorted(candidate_indices)
    i = 0

    while i < len(sorted_indices):
        start_idx = max(0, sorted_indices[i] - window_size)
        end_idx = sorted_indices[i] + window_size
        targets_in_window = [sorted_indices[i]]

        # Extend window to include adjacent candidates
        j = i + 1
        while j < len(sorted_indices):
            next_start = max(0, sorted_indices[j] - window_size)
            if next_start <= end_idx:
                end_idx = sorted_indices[j] + window_size
                targets_in_window.append(sorted_indices[j])
                j += 1
            else:
                break

        merged.append((start_idx, end_idx, targets_in_window))
        i = j

    return merged


def _sparse_motion_analysis(cap, fps, total_frames):
    """Phase 1: Sparse sampling with DIS optical flow."""
    sample_interval = max(1, int(fps * 0.5))
    sparse_samples = []
    dis_flow = _create_dis_flow()
    current_idx = 0
    prev_gray = None

    while True:
        if current_idx > 0:
            steps_to_skip = sample_interval - 1
            if steps_to_skip > 0:
                current_idx = _advance_cap_to_frame(cap, current_idx, current_idx + steps_to_skip)
        ret, frame = cap.read()
        if not ret:
            break

        small = _resize_for_flow(frame, long_edge=320)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        motion_mag = 0.0
        if prev_gray is not None:
            if dis_flow is not None:
                flow = dis_flow.calc(prev_gray, gray, None)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 2, 5, 1.2, 0)
            motion_mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))

        sparse_samples.append({"idx": current_idx, "motion": motion_mag, "hist": _calculate_histogram(small)})
        prev_gray = gray
        current_idx += 1

    return sparse_samples


def _adaptive_frame_selection(sparse_samples, fps, max_frames):
    """Phase 2: Adaptive threshold allocation with deduplication."""
    motions = [s["motion"] for s in sparse_samples[1:]]

    if not motions:
        return [sparse_samples[0]["idx"]]

    # Calculate adaptive threshold
    static_floor = 1.0
    total_motion = sum(motions)
    estimated_step = total_motion / max_frames if max_frames > 0 else total_motion
    step_threshold = max(static_floor * 5.0, estimated_step)

    # Select frames based on accumulated motion
    candidate_indices = [sparse_samples[0]["idx"]]
    selected_hists = [sparse_samples[0]["hist"]]
    current_accum = 0.0
    last_selected_idx = sparse_samples[0]["idx"]

    for i in range(1, len(sparse_samples)):
        s = sparse_samples[i]
        effective_motion = s["motion"] if s["motion"] >= static_floor else 0.0
        current_accum += effective_motion

        time_gap = s["idx"] - last_selected_idx
        should_select = (current_accum >= step_threshold) or (time_gap > (4.0 * fps))

        if should_select:
            is_duplicate = any(_calculate_hist_similarity(s["hist"], h) > 0.999 for h in selected_hists)
            if not is_duplicate:
                candidate_indices.append(s["idx"])
                selected_hists.append(s["hist"])
                current_accum = 0.0
                last_selected_idx = s["idx"]

    # Always check last frame
    if sparse_samples[-1]["idx"] != candidate_indices[-1]:
        last_hist = sparse_samples[-1]["hist"]
        if not any(_calculate_hist_similarity(last_hist, h) > 0.999 for h in selected_hists):
            candidate_indices.append(sparse_samples[-1]["idx"])

    return sorted(list(set(candidate_indices)))


def _enforce_frame_constraints(candidate_indices, sparse_samples, min_frames, max_frames):
    """Enforce min/max frame constraints."""
    if len(candidate_indices) < min_frames:
        needed = min_frames - len(candidate_indices)
        all_indices = [s["idx"] for s in sparse_samples]
        extras = np.linspace(0, len(all_indices) - 1, needed + 2)[1:-1]
        candidate_indices.extend([all_indices[int(e)] for e in extras])
        candidate_indices = sorted(list(set(candidate_indices)))

    if len(candidate_indices) > max_frames:
        indices_to_keep = np.linspace(0, len(candidate_indices) - 1, max_frames)
        candidate_indices = [candidate_indices[int(round(i))] for i in indices_to_keep]

    return candidate_indices


def _read_window_frames(cap, merged_windows, total_frames):
    """Read all frames from merged windows."""
    all_frames = []
    current_pos = -1

    for window_idx, (window_start, window_end, _) in enumerate(merged_windows):
        current_pos = _advance_cap_to_frame(cap, current_pos, window_start)
        for idx in range(window_start, min(window_end + 1, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append((window_idx, idx, frame))
            current_pos = idx + 1

    return all_frames


def _compute_clarity_parallel(all_frames):
    """Parallel clarity calculation."""

    def _compute(item):
        window_idx, frame_idx, frame = item
        clarity_frame = _resize_for_clarity(frame, long_edge=480)
        gray = cv2.cvtColor(clarity_frame, cv2.COLOR_BGR2GRAY)
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
        return (window_idx, frame_idx, frame, clarity)

    with ThreadPoolExecutor(max_workers=min(8, len(all_frames) or 1)) as ex:
        return list(ex.map(_compute, all_frames))


def _select_best_frames(clarity_results, merged_windows, candidate_indices, search_window_size=3):
    """Select best frame for each candidate based on clarity."""
    # Group by window
    window_frames = {}
    for window_idx, frame_idx, frame, clarity in clarity_results:
        if window_idx not in window_frames:
            window_frames[window_idx] = []
        window_frames[window_idx].append((frame_idx, frame, clarity))

    # Select best frame for each target
    target_to_best = {}
    for window_idx, (_, _, targets) in enumerate(merged_windows):
        frames = window_frames.get(window_idx, [])
        for target_idx in targets:
            candidates = [(idx, f, c) for idx, f, c in frames if abs(idx - target_idx) <= search_window_size]
            if candidates:
                best_idx, best_frame, _ = max(candidates, key=lambda x: x[2])
                target_to_best[target_idx] = (best_idx, best_frame)
            elif frames:
                closest = min(frames, key=lambda x: abs(x[0] - target_idx))
                target_to_best[target_idx] = (closest[0], closest[1])

    return target_to_best


def _save_frames_parallel(target_to_best, candidate_indices, save_directory):
    """Parallel frame saving."""
    path_frame_list = []
    final_indices = []

    for target_idx in sorted(candidate_indices):
        if target_idx in target_to_best:
            best_idx, best_frame = target_to_best[target_idx]
            final_indices.append(best_idx)
            path_frame_list.append((os.path.join(save_directory, f"frame_{len(path_frame_list):06d}.jpg"), best_frame))

    def _write(p_f):
        cv2.imwrite(p_f[0], p_f[1])
        return p_f[0]

    with ThreadPoolExecutor(max_workers=min(8, len(path_frame_list) or 1)) as ex:
        paths = list(ex.map(_write, path_frame_list))

    return final_indices, paths


def video_to_image_frames_new(
    input_video_path,
    save_directory=None,
    min_frames=1,
    max_frames=64,
    fallback_fps=1,
):
    """
    Motion-aware frame extraction with local clarity refinement.

    Strategy:
    1. Sparse sampling (~0.5s) with DIS optical flow
    2. Adaptive threshold allocation based on motion
    3. Local clarity refinement (±3 frames) to avoid blur
    """
    if save_directory is None:
        raise ValueError("save_directory must be provided")

    max_frames = int(np.clip(max_frames, 1, 64))
    min_frames = int(np.clip(min_frames, 1, max_frames))

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video {input_video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or fallback_fps or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_start = time.perf_counter()

    # Phase 1: Sparse motion analysis
    sparse_samples = _sparse_motion_analysis(cap, fps, total_frames)
    cap.release()

    t_phase1 = time.perf_counter()
    print(f"[Timing] Phase 1 (Sparse Flow): {t_phase1 - t_start:.3f}s, Samples: {len(sparse_samples)}")

    if not sparse_samples:
        return []

    # Phase 2: Adaptive frame selection
    candidate_indices = _adaptive_frame_selection(sparse_samples, fps, max_frames)
    candidate_indices = _enforce_frame_constraints(candidate_indices, sparse_samples, min_frames, max_frames)

    # Phase 3: Local clarity refinement
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return []

    t_phase3_start = time.perf_counter()
    search_window_size = 3
    merged_windows = _merge_search_windows(candidate_indices, window_size=search_window_size)

    # Read frames
    t_read_start = time.perf_counter()
    all_frames = _read_window_frames(cap, merged_windows, total_frames)
    cap.release()
    t_read_end = time.perf_counter()

    # Parallel clarity calculation
    t_clarity_start = time.perf_counter()
    clarity_results = _compute_clarity_parallel(all_frames)
    t_clarity_end = time.perf_counter()

    # Select best frames
    target_to_best = _select_best_frames(clarity_results, merged_windows, candidate_indices, search_window_size)

    # Parallel save
    t_save_start = time.perf_counter()
    final_indices, extracted_paths = _save_frames_parallel(target_to_best, candidate_indices, save_directory)
    t_save_end = time.perf_counter()

    t_phase3_end = time.perf_counter()
    print(f"[Timing] Phase 3 (Clarity Refinement + Save): {t_phase3_end - t_phase3_start:.3f}s")
    print(f"  - Read frames: {t_read_end - t_read_start:.3f}s")
    print(f"  - Parallel clarity: {t_clarity_end - t_clarity_start:.3f}s")
    print(f"  - Parallel save: {t_save_end - t_save_start:.3f}s, Saved: {len(extracted_paths)}")

    # Save metadata
    try:
        meta = {"frame_indices": final_indices, "frame_times": [i / fps for i in final_indices], "fps": fps, "algorithm": "sparse_dis_clarity_refined"}
        with open(os.path.join(save_directory, "frame_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        with open(os.path.join(save_directory, "frame_metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "time_sec", "motion", "selected"])
            for s in sparse_samples:
                writer.writerow([s["idx"], s["idx"] / fps, s["motion"], 1 if s["idx"] in final_indices else 0])
    except Exception:
        pass

    print(f"Extracted {len(extracted_paths)} frames using DIS flow + local clarity refinement.")
    return extracted_paths
