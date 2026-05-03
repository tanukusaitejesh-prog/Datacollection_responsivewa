from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import contextlib
from .supabase_db import push_metadata_to_supabase, upload_file_to_supabase
from .mongo_db import push_metadata_to_mongo

from .landmarks25 import (
    CUSTOM_25_CSV_COLUMNS,
    CUSTOM_25_MAPPING_METADATA,
    convert_33_to_custom_25,
    flatten_custom_25_for_csv,
)
import pandas as pd
from .full_export import flatten_full_capture, generate_full_csv_headers
from .pipeline import PipelineConfig, preprocess_pose_capture
from .render import render_skeleton_video
from .schemas import UploadPayload

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_JSON_DIR = DATA_DIR / "raw_json"
RAW_PTH_DIR = DATA_DIR / "raw_pth"
PROCESSED_PTH_DIR = DATA_DIR / "processed_pth"
RENDER_DIR = DATA_DIR / "renders"
PROCESSED_25_CSV_DIR = DATA_DIR / "processed_25_csv"
PROCESSED_FULL_CSV_DIR = DATA_DIR / "processed_full_csv"
PROCESSED_FULL_XLSX_DIR = DATA_DIR / "processed_full_xlsx"
PROCESSED_25_NPY_DIR = DATA_DIR / "processed_25_npy"
PROCESSED_33_NPY_DIR = DATA_DIR / "processed_33_npy"
PROCESSED_25_META_DIR = DATA_DIR / "processed_25_meta"

for folder in [
    RAW_JSON_DIR,
    RAW_PTH_DIR,
    PROCESSED_PTH_DIR,
    RENDER_DIR,
    PROCESSED_25_CSV_DIR,
    PROCESSED_FULL_CSV_DIR,
    PROCESSED_FULL_XLSX_DIR,
    PROCESSED_25_NPY_DIR,
    PROCESSED_33_NPY_DIR,
    PROCESSED_25_META_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown

app = FastAPI(title="Pose Capture Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _slugify_session(session_id: str | None) -> str:
    if not session_id:
        return "capture"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", session_id).strip("-")
    return slug[:48] if slug else "capture"


def _capture_id(session_id: str | None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{_slugify_session(session_id)}-{now}-{suffix}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_capture(payload: UploadPayload) -> dict[str, object]:
    capture_id = _capture_id(payload.meta.session_id)
    raw_json_path = RAW_JSON_DIR / f"{capture_id}.json"
    raw_pth_path = RAW_PTH_DIR / f"{capture_id}.pth"
    processed_pth_path = PROCESSED_PTH_DIR / f"{capture_id}.pth"
    render_path = RENDER_DIR / f"{capture_id}.mp4"
    processed_25_csv_path = PROCESSED_25_CSV_DIR / f"{capture_id}.csv"
    processed_full_csv_path = PROCESSED_FULL_CSV_DIR / f"{capture_id}.csv"
    processed_full_xlsx_path = PROCESSED_FULL_XLSX_DIR / f"{capture_id}.xlsx"
    processed_25_npy_path = PROCESSED_25_NPY_DIR / f"{capture_id}.npy"
    processed_33_npy_path = PROCESSED_33_NPY_DIR / f"{capture_id}.npy"
    processed_25_meta_path = PROCESSED_25_META_DIR / f"{capture_id}.json"

    try:
        keypoints_np = np.asarray(payload.keypoints, dtype=np.float32)
        timestamps_np = np.asarray(payload.timestamps, dtype=np.float64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid numeric payload: {exc}") from exc

    raw_json_path.write_text(
        json.dumps(payload.model_dump(), separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )

    torch.save(
        {
            "keypoints": torch.from_numpy(keypoints_np),
            "timestamps": torch.from_numpy(timestamps_np),
            "meta": payload.meta.model_dump(),
        },
        raw_pth_path,
    )

    config = PipelineConfig(
        target_fps=float(payload.meta.fps_nominal),
        smoothing_alpha=0.35,
        visibility_threshold=0.5,
    )
    processed = preprocess_pose_capture(keypoints_np, timestamps_np, config)

    torch.save(
        {
            "keypoints": torch.from_numpy(processed.keypoints.astype(np.float32)),
            "timestamps": torch.from_numpy(processed.timestamps.astype(np.float64)),
            "meta": {
                **payload.meta.model_dump(),
                "processing": processed.processing_meta,
            },
        },
        processed_pth_path,
    )

    try:
        keypoints_25 = convert_33_to_custom_25(keypoints_np, invert_xy=True)
        keypoints_25_flat = flatten_custom_25_for_csv(keypoints_25)

        np.save(processed_25_npy_path, keypoints_25.astype(np.float32))
        if processed.screening_npy is not None:
            np.save(processed_33_npy_path, processed.screening_npy)

        np.savetxt(
            processed_25_csv_path,
            keypoints_25_flat,
            delimiter=",",
            header=",".join(CUSTOM_25_CSV_COLUMNS),
            comments="",
        )

        # Generate FULL CSV (1600+ columns)
        full_data = flatten_full_capture(payload, keypoints_25, timestamps_np)
        full_headers = generate_full_csv_headers()
        np.savetxt(
            processed_full_csv_path,
            full_data,
            delimiter=",",
            header=",".join(full_headers),
            fmt="%s",
            comments="",
        )

        # Generate FULL XLSX
        df = pd.DataFrame(full_data, columns=full_headers)
        df.to_excel(processed_full_xlsx_path, index=False)

        timestamps_start = (
            float(timestamps_np[0]) if timestamps_np.shape[0] > 0 else None
        )
        timestamps_end = (
            float(timestamps_np[-1]) if timestamps_np.shape[0] > 0 else None
        )
        conversion_meta = {
            "capture_id": capture_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "type": "raw_mobile_upload",
                "keypoints_shape": list(keypoints_np.shape),
                "timestamps_shape": list(timestamps_np.shape),
                "timestamps_start_raw": timestamps_start,
                "timestamps_end_raw": timestamps_end,
                "raw_json_path": str(raw_json_path),
                "raw_pth_path": str(raw_pth_path),
            },
            "conversion": {
                "name": "raw_mediapipe33_to_custom25",
                "xy_sign_inverted": True,
                "target_shape": list(keypoints_25.shape),
                "csv_columns": CUSTOM_25_CSV_COLUMNS,
                "mapping": CUSTOM_25_MAPPING_METADATA,
            },
            "capture_meta": payload.meta.model_dump(),
            "files": {
                "csv_25_path": str(processed_25_csv_path),
                "csv_full_path": str(processed_full_csv_path),
                "xlsx_full_path": str(processed_full_xlsx_path),
                "npy_25_path": str(processed_25_npy_path),
                "npy_33_path": str(processed_33_npy_path),
            },
        }
        processed_25_meta_path.write_text(
            json.dumps(conversion_meta, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Custom 25-landmark export failed: {exc}",
        ) from exc

    try:
        render_skeleton_video(
            keypoints=processed.keypoints,
            output_path=render_path,
            fps=float(payload.meta.fps_nominal),
            canvas_size=720,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Render failed: {exc}") from exc


    # Supabase Integration (Parallel)
    try:
        storage_paths = {}
        # Upload key files to Supabase Storage
        # Map: local_path -> remote_name
        files_to_upload = {
            render_path: f"{capture_id}/skeleton.mp4",
            processed_25_csv_path: f"{capture_id}/data_25.csv",
            processed_full_csv_path: f"{capture_id}/data_full.csv",
            processed_full_xlsx_path: f"{capture_id}/data_full.xlsx",
            processed_33_npy_path: f"{capture_id}/data_33_screening.npy",
        }

        for local_p, remote_n in files_to_upload.items():
            public_url = await upload_file_to_supabase(local_p, remote_n)
            if public_url:
                storage_paths[remote_n.split("/")[-1]] = public_url

        # Push metadata to Supabase Table
        supabase_payload = {
            "capture_id": capture_id,
            "meta": conversion_meta,
            "storage_paths": storage_paths
        }
        await push_metadata_to_supabase(supabase_payload)
        print(f"Capture {capture_id} data pushed to Supabase.")
    except Exception as exc:
        print(f"Failed to push to Supabase: {exc}")

    # MongoDB Integration (Local)
    try:
        mongo_payload = {
            "capture_id": capture_id,
            "meta": conversion_meta,
            "storage_paths": storage_paths if 'storage_paths' in dir() else {},
        }
        await push_metadata_to_mongo(mongo_payload)
        print(f"Capture {capture_id} data pushed to local MongoDB.")
    except Exception as exc:
        print(f"Failed to push to MongoDB: {exc}")

    return {
        "status": "ok",
        "capture_id": capture_id,
        "raw_json_path": str(raw_json_path),
        "raw_pth_path": str(raw_pth_path),
        "processed_pth_path": str(processed_pth_path),
        "render_path": str(render_path),
        "processed_25_csv_path": str(processed_25_csv_path),
        "processed_full_csv_path": str(processed_full_csv_path),
        "processed_full_xlsx_path": str(processed_full_xlsx_path),
        "processed_25_npy_path": str(processed_25_npy_path),
        "processed_33_npy_path": str(processed_33_npy_path),
        "processed_25_meta_path": str(processed_25_meta_path),
        "frames_in": int(keypoints_np.shape[0]),
        "frames_out": int(processed.keypoints.shape[0]),
    }
