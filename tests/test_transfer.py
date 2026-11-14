"""Results copied between Windows and Linux retain verifiable receipts."""
import json

from p4.common import sha256_file
from p4.experiment import stage_receipt_valid


def test_receipt_accepts_different_python_path_but_rejects_changed_stage(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_text('{"value":1}')
    receipt = tmp_path / "receipt.json"
    command = ["/opt/project4/bin/python", "-m", "p4.evaluate", "renders", "--dataset", "data/prepared/Dl3dv-2"]
    saved = [r"C:\project\.venv\Scripts\python.exe", "-m", "p4.evaluate", "renders", "--dataset", r"data\prepared\Dl3dv-2"]
    receipt.write_text(json.dumps({"command": saved, "signature": "frozen", "artifacts": {artifact.as_posix(): sha256_file(artifact)}}))
    assert stage_receipt_valid(receipt, command, "frozen")
    assert not stage_receipt_valid(receipt, command + ["--no-lpips"], "frozen")
    artifact.write_text('{"value":2}')
    assert not stage_receipt_valid(receipt, command, "frozen")
