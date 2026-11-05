#!/usr/bin/env python3
"""
Download inference artifacts from a private S3 prefix into the app working directory.

Uses default credential chain (ECS task role on Fargate; locally ~/.aws or env vars). Intended to run before uvicorn.

Environment:
  MODEL_S3_URI          s3://bucket/prefix (lists objects under prefix; trailing slash optional)
  or
  MODEL_S3_BUCKET       bucket name
  MODEL_S3_PREFIX       key prefix (use empty string for bucket root: downloads allowlisted basenames only)

  MODEL_SYNC_DEST       destination directory (default: current working directory, or /app in Docker)
  MODEL_S3_REGION       optional AWS region for the S3 client
  MODEL_S3_EXTRA_FILES  optional comma-separated extra basenames to download (in addition to defaults)

Default: only downloads known inference basenames (model.pt, model.pkl, model_config.pkl, …).
Set MODEL_S3_ALLOW_ANY=1 to download every object under the prefix (basename written to dest; collisions overwrite).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

DEFAULT_BASENAMES = frozenset(
    {
        "model.pt",
        "model.pkl",
        "model_config.pkl",
        "feature_scaler.pkl",
        "tfidf_vectorizer.pkl",
    }
)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    u = uri.strip()
    if not u.lower().startswith("s3://"):
        raise ValueError("MODEL_S3_URI must start with s3://")
    rest = u[5:].lstrip("/")
    if not rest:
        raise ValueError("MODEL_S3_URI must include a bucket name")
    if "/" not in rest:
        return rest, ""
    bucket, prefix = rest.split("/", 1)
    prefix = prefix.strip("/")
    return bucket, prefix


def _resolve_bucket_prefix() -> tuple[str, str] | None:
    uri = (os.environ.get("MODEL_S3_URI") or "").strip()
    if uri:
        return _parse_s3_uri(uri)
    bucket = (os.environ.get("MODEL_S3_BUCKET") or "").strip()
    if not bucket:
        return None
    if "MODEL_S3_PREFIX" not in os.environ:
        print(
            "sync_model_from_s3: MODEL_S3_BUCKET is set but MODEL_S3_PREFIX is unset. "
            "Set MODEL_S3_PREFIX (use empty string for bucket root) or use MODEL_S3_URI.",
            file=sys.stderr,
        )
        sys.exit(1)
    prefix = (os.environ.get("MODEL_S3_PREFIX") or "").strip().strip("/")
    return bucket, prefix


def _allowed_basenames() -> frozenset[str]:
    names = set(DEFAULT_BASENAMES)
    extra = (os.environ.get("MODEL_S3_EXTRA_FILES") or "").strip()
    if extra:
        for part in extra.split(","):
            p = part.strip()
            if p:
                names.add(p)
    return frozenset(names)


def _should_download_key(key: str, allow_any: bool, allowed: frozenset[str]) -> bool:
    if key.endswith("/"):
        return False
    base = Path(key).name
    if not base or base.startswith("."):
        return False
    if allow_any:
        return True
    return base in allowed


def _download_one(
    client, bucket: str, key: str, dest: Path
) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    base = Path(key).name
    out_path = dest / base
    print(f"sync_model_from_s3: downloading s3://{bucket}/{key} -> {out_path}")
    client.download_file(bucket, key, str(out_path))
    return True


def main() -> int:
    resolved = _resolve_bucket_prefix()
    if resolved is None:
        print("sync_model_from_s3: MODEL_S3_URI / MODEL_S3_BUCKET not set; skipping.")
        return 0

    bucket, prefix = resolved
    dest = Path(os.environ.get("MODEL_SYNC_DEST") or os.getcwd()).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    allow_any = (os.environ.get("MODEL_S3_ALLOW_ANY") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    allowed = _allowed_basenames()

    kwargs: dict = {}
    region = (os.environ.get("MODEL_S3_REGION") or "").strip()
    if region:
        kwargs["region_name"] = region

    client = boto3.client("s3", **kwargs)

    downloaded = 0
    try:
        if prefix:
            list_prefix = prefix.rstrip("/") + "/"
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key") or ""
                    if not _should_download_key(key, allow_any, allowed):
                        continue
                    base = Path(key).name
                    out_path = dest / base
                    print(f"sync_model_from_s3: downloading s3://{bucket}/{key} -> {out_path}")
                    client.download_file(bucket, key, str(out_path))
                    downloaded += 1
        else:
            if allow_any:
                print(
                    "sync_model_from_s3: MODEL_S3_ALLOW_ANY=1 with an empty prefix is not supported "
                    "(would require a full bucket listing). Set a prefix in MODEL_S3_URI.",
                    file=sys.stderr,
                )
                return 1
            for base in sorted(allowed):
                key = base
                if _download_one(client, bucket, key, dest):
                    downloaded += 1
    except NoCredentialsError:
        print(
            "sync_model_from_s3: no AWS credentials (Unable to locate credentials). "
            "On ECS attach an IAM task role with S3 permissions. "
            "Locally: run `aws sso login` / configure ~/.aws, or pass keys via env (dev only). "
            "In Docker: mount credentials, e.g. -v ~/.aws:/home/appuser/.aws:ro -e AWS_PROFILE=your-profile "
            "(container user is appuser; HOME=/home/appuser).",
            file=sys.stderr,
        )
        return 1
    except (ClientError, BotoCoreError) as e:
        print(f"sync_model_from_s3: S3 error: {e}", file=sys.stderr)
        return 1

    if downloaded == 0:
        loc = f"s3://{bucket}/{prefix}/…" if prefix else f"s3://{bucket}/<basename>"
        print(
            f"sync_model_from_s3: no matching objects ({loc}, allow_any={allow_any}).",
            file=sys.stderr,
        )
        return 1

    print(f"sync_model_from_s3: done ({downloaded} file(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
