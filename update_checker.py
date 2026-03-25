"""
HuggingFace model update checker for PushToTalk.
Checks for new revisions of Whisper models on startup.
"""

import json
import os
import logging
import urllib.request
from datetime import datetime

logger = logging.getLogger("ptt")

VERSIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_versions.json")
HF_API = "https://huggingface.co/api/models/{}"


def check_model_updates(model_ids):
    """Check HuggingFace for updates to the given models.

    Returns dict: {model_id: {"updated": bool, "last_modified": str, "error": str|None}}
    First run stores baseline (updated=False).
    """
    stored = _load_versions()
    results = {}

    for model_id in model_ids:
        try:
            url = HF_API.format(model_id)
            req = urllib.request.Request(url, headers={"User-Agent": "PushToTalk/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            new_sha = data.get("sha", "")
            last_modified = data.get("lastModified", "")

            old_sha = stored.get(model_id, {}).get("sha", "")
            has_update = old_sha != "" and old_sha != new_sha

            if not old_sha:
                short = model_id.split("/")[-1]
                logger.info(f"HF baseline: {short} (sha: {new_sha[:8]})")

            results[model_id] = {
                "updated": has_update,
                "sha": new_sha,
                "last_modified": last_modified,
                "error": None,
            }

            stored[model_id] = {
                "sha": new_sha,
                "last_modified": last_modified,
                "checked": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.debug(f"HF check failed for {model_id}: {e}")
            results[model_id] = {"updated": False, "error": str(e)}

    _save_versions(stored)
    return results


def _load_versions():
    if os.path.exists(VERSIONS_PATH):
        try:
            with open(VERSIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_versions(data):
    try:
        with open(VERSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.debug(f"Failed to save versions: {e}")
