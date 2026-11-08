"""Tests for scripts/sync_model_from_s3.py (boto3 mocked)."""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "sync_model_from_s3", _SCRIPTS / "sync_model_from_s3.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_mod)


class TestSyncModelFromS3(unittest.TestCase):
    def test_skip_when_not_configured(self):
        with patch.dict(os.environ):
            for k in ("MODEL_S3_URI", "MODEL_S3_BUCKET", "MODEL_S3_PREFIX", "MODEL_SYNC_DEST", "PARAGRAPH_MODEL_S3_URI"):
                os.environ.pop(k, None)
            self.assertEqual(_mod.main(), 0)

    def test_prefix_lists_and_downloads(self):
        mock_client = MagicMock()
        page = {
            "Contents": [
                {"Key": "models/prod/model_config.pkl"},
                {"Key": "models/prod/model.pt"},
                {"Key": "models/prod/README.md"},
            ]
        }
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [page]
        mock_client.get_paginator.return_value = mock_paginator

        tmp = Path(self._temp_dest()).resolve()
        env = {
            "MODEL_S3_URI": "s3://my-bucket/models/prod",
            "MODEL_SYNC_DEST": str(tmp),
        }
        with patch.dict(os.environ, env, clear=True):
            mock_boto = MagicMock()
            mock_boto.client.return_value = mock_client
            with patch.object(_mod, "boto3", mock_boto):
                self.assertEqual(_mod.main(), 0)

        mock_client.download_file.assert_any_call("my-bucket", "models/prod/model_config.pkl", str(tmp / "model_config.pkl"))
        mock_client.download_file.assert_any_call("my-bucket", "models/prod/model.pt", str(tmp / "model.pt"))
        downloaded_keys = {c[0][1] for c in mock_client.download_file.call_args_list}
        self.assertNotIn("models/prod/README.md", downloaded_keys)

    def test_bucket_root_tries_allowlisted_keys(self):
        mock_client = MagicMock()

        def head_side_effect(Bucket, Key):
            if Key != "model.pkl":
                from botocore.exceptions import ClientError

                err = ClientError({"Error": {"Code": "404"}}, "HeadObject")
                raise err

        mock_client.head_object.side_effect = head_side_effect

        tmp = Path(self._temp_dest()).resolve()
        env = {
            "MODEL_S3_BUCKET": "b",
            "MODEL_S3_PREFIX": "",
            "MODEL_SYNC_DEST": str(tmp),
        }
        with patch.dict(os.environ, env, clear=True):
            mock_boto = MagicMock()
            mock_boto.client.return_value = mock_client
            with patch.object(_mod, "boto3", mock_boto):
                self.assertEqual(_mod.main(), 0)

        mock_client.download_file.assert_called_once_with("b", "model.pkl", str(tmp / "model.pkl"))

    def paragraph_client(self, keys, fail_on=None):
        client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": [{"Key": key} for key in keys]}]
        client.get_paginator.return_value = paginator

        def download(bucket, key, path):
            if key == fail_on:
                from botocore.exceptions import ClientError

                raise ClientError({"Error": {"Code": "500"}}, "GetObject")
            Path(path).write_text(key)

        client.download_file.side_effect = download
        return client

    def sync_paragraph_model(self, client, dest):
        env = {"PARAGRAPH_MODEL_S3_URI": "s3://b/paragraph/scibert-full-1108/", "PARAGRAPH_MODEL_SYNC_DEST": str(dest)}
        with patch.dict(os.environ, env, clear=True):
            mock_boto = MagicMock()
            mock_boto.client.return_value = client
            with patch.object(_mod, "boto3", mock_boto):
                return _mod.main()

    def test_paragraph_model_keeps_its_folder_layout(self):
        dest = Path(self._temp_dest()) / "paragraph_model"
        client = self.paragraph_client([
            "paragraph/scibert-full-1108/model_config.pkl",
            "paragraph/scibert-full-1108/calibrator.pkl",
            "paragraph/scibert-full-1108/model/config.json",
            "paragraph/scibert-full-1108/model/model.safetensors",
        ])

        self.assertEqual(self.sync_paragraph_model(client, dest), 0)

        self.assertEqual(
            sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()),
            ["calibrator.pkl", "model/config.json", "model/model.safetensors", "model_config.pkl"],
        )

    def test_paragraph_model_failed_download_leaves_the_old_model_in_place(self):
        dest = Path(self._temp_dest()) / "paragraph_model"
        dest.mkdir()
        (dest / "model_config.pkl").write_text("old")
        client = self.paragraph_client(
            ["paragraph/scibert-full-1108/model_config.pkl", "paragraph/scibert-full-1108/model/model.safetensors"],
            fail_on="paragraph/scibert-full-1108/model/model.safetensors",
        )

        self.assertEqual(self.sync_paragraph_model(client, dest), 1)

        self.assertEqual((dest / "model_config.pkl").read_text(), "old")
        self.assertEqual([p.name for p in dest.parent.iterdir()], ["paragraph_model"])

    def test_paragraph_model_refuses_keys_that_climb_out_of_the_folder(self):
        dest = Path(self._temp_dest()) / "paragraph_model"
        client = self.paragraph_client(["paragraph/scibert-full-1108/model_config.pkl", "paragraph/scibert-full-1108/../../evil.py"])

        self.assertEqual(self.sync_paragraph_model(client, dest), 1)
        self.assertFalse(dest.exists())

    def test_paragraph_model_without_its_config_is_refused(self):
        dest = Path(self._temp_dest()) / "paragraph_model"
        client = self.paragraph_client(["paragraph/scibert-full-1108/model/config.json"])

        self.assertEqual(self.sync_paragraph_model(client, dest), 1)
        self.assertFalse(dest.exists())

    def _temp_dest(self) -> str:
        import tempfile

        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


if __name__ == "__main__":
    unittest.main()
