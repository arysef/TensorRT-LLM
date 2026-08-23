# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adjudicate the registered protected-regression run, per test, from evidence.

The Stage 3 regression criterion is not "pytest exited 0" --- it cannot be, on
this machine. The dev container the whole bring-up builds and measures in is
created without ``CAP_SYS_PTRACE``, and every test that starts an MPI pool dies
in ``pidfd_getfd`` with ``Operation not permitted``. That is a property of the
container, not of any DeepSeek-V4 change. The criterion is therefore:

* every failure is *demonstrably* that environment fault, judged from the
  failure's own traceback rather than from a list of ids someone forgave;
* no failure is new against the registered baseline;
* no test the Hopper gates require is skipped or failing;
* Blackwell dispatch is covered statically and Blackwell *runtime* is reported
  ``Not measured`` rather than inferred.

What makes this an adjudication rather than a waiver is that
:func:`classify_failure` reads the failure text. A node id sitting in the
baseline whose failure no longer matches the environment signature is reported
``genuine`` and fails the report; the baseline can only ever *narrow* what
counts as expected, never widen it.

Two inputs, and each answers the half it is good for. The registered command's
own terminal log carries the counts, the exit code and every failure's
traceback, so failures are classified from the registered run itself rather
than from a re-run that might not be the same run. Skips do not carry their
reason in that log, so the required-H100 half is driven by a focused
``--junitxml`` pass over just those cases --- which is also the only half where
a skip is a finding at all.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(_HERE, "manifests", "regression_baseline.json")

#: The pytest terminal summary, e.g. ``74 failed, 407 passed, 1356 skipped,
#: 3 warnings in 512.34s``. ``in <n>s`` anchors it so a traceback that happens
#: to quote "3 passed" cannot be mistaken for the summary.
_COUNT_RE = re.compile(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)")
_SUMMARY_RE = re.compile(r"\bin \d+(\.\d+)?s")

#: ``_____________ test_name _____________`` opens each traceback in the
#: FAILURES / ERRORS section. The underscore padding shrinks to a single
#: character once the test's name plus its parametrisation fills the terminal
#: width, which is the common case here, so the run length is not constrained.
#: The intra-traceback ``_ _ _ _`` separator matches this shape too and is
#: rejected by :func:`_block_name` rather than by a narrower pattern.
_BLOCK_RE = re.compile(r"^_+ (\S.*?) _+$")
#: Any ``===== title =====`` banner. Only FAILURES/ERRORS *opens* a traceback
#: section; every other banner closes it. Matching the closers by name would
#: silently swallow whichever section pytest happens to print next --- the
#: durations report, for instance, when a run emits no warnings.
_SECTION_RE = re.compile(r"^=+ (.*?) =+$")
_TRACEBACK_SECTIONS = ("FAILURES", "ERRORS")


def parse_pytest_log(path: str) -> dict[str, Any]:
    """Counts, exit code and per-failure tracebacks from a ``pytest -q`` log.

    The tracebacks are what the classification reads, so they are taken from
    the registered command's own output. ``FAILED``/``ERROR`` summary lines
    supply the node ids, and the traceback blocks --- keyed by the test name
    pytest heads each block with --- supply the text.
    """
    with open(path, errors="replace") as f:
        text = f.read()

    counts: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not _SUMMARY_RE.search(stripped):
            continue
        found = _COUNT_RE.findall(stripped)
        if found:
            counts = {("error" if k.startswith("error") else k): int(n) for n, k in found}

    exit_match = re.search(r"^EXIT=(\d+)$", text, re.MULTILINE)
    summary = [
        (outcome.lower(), node_id)
        for outcome, node_id in re.findall(r"^(FAILED|ERROR) (\S+)", text, re.MULTILINE)
    ]
    occurrences = _traceback_blocks(text)

    claimed: set[int] = set()
    failures = []
    for outcome, node_id in summary:
        text_for, source = _claim_block(node_id, occurrences, claimed)
        failures.append(
            {
                "node_id": node_id,
                "outcome": outcome,
                "message": _summary_message(text, node_id),
                "text": text_for,
                # How the traceback was tied to this node id, so an
                # unresolved one is visible rather than silently empty.
                "traceback_source": source,
            }
        )

    return {
        "path": path,
        "counts": counts,
        "exit_code": None if exit_match is None else int(exit_match.group(1)),
        "failures": failures,
        "failed_ids": sorted(n for o, n in summary if o == "failed"),
        "error_ids": sorted(n for o, n in summary if o == "error"),
    }


def _summary_message(text: str, node_id: str) -> str:
    match = re.search(rf"^(?:FAILED|ERROR) {re.escape(node_id)}(?: - (.*))?$", text, re.MULTILINE)
    return (match.group(1) or "") if match else ""


def _block_name(captured: str) -> str | None:
    """The test a traceback banner names, or ``None`` if it is not a banner.

    ``_ _ _ _ _`` separates frames *inside* a traceback and has the same shape
    as a banner once the padding shrinks to one underscore, so a line made only
    of underscores and spaces is rejected: treating one as a banner would split
    a single failure into fragments and lose the exception line.
    """
    name = re.sub(r"^ERROR at (?:setup|teardown) of ", "", captured).strip()
    return None if not name or set(name) <= {"_", " "} else name


def _traceback_blocks(text: str) -> list[dict[str, str]]:
    """The FAILURES / ERRORS section as an ordered list of banner occurrences.

    A list, not a ``name -> text`` mapping. pytest heads each block with the
    test's *leaf* name, which is not unique across files: two files may each
    define ``test_same``. Merging them under one key gave both failures both
    tracebacks, so an assertion failure inherited its namesake's ``pidfd_getfd``
    text and was classified as the container fault. Occurrences are kept
    separate here and matched one-to-one in :func:`_claim_block`.
    """
    occurrences: list[dict[str, Any]] = []
    lines: list[str] = []
    in_section = False
    for line in text.splitlines():
        section = _SECTION_RE.match(line.strip())
        if section:
            in_section = section.group(1).strip() in _TRACEBACK_SECTIONS
            lines = []
            continue
        if not in_section:
            continue
        header = _BLOCK_RE.match(line.strip())
        name = None if header is None else _block_name(header.group(1))
        if name:
            lines = []
            occurrences.append({"name": name, "lines": lines})
            continue
        lines.append(line)
    return [{"name": o["name"], "text": "\n".join(o["lines"])} for o in occurrences]


def _claim_block(
    node_id: str, occurrences: list[dict[str, str]], claimed: set[int]
) -> tuple[str, str]:
    """Take the one traceback that belongs to ``node_id``, or take none.

    Resolution narrows, and stops rather than guessing:

    1. banner name equals the node id's leaf name, among blocks not already
       taken by an earlier summary line;
    2. if that is still ambiguous, the block's own traceback must mention the
       node id's file, which pytest prints in every frame;
    3. anything still ambiguous --- or already claimed --- yields no text.

    Step 3 is the important one. Returning nothing makes the failure
    ``genuine`` with an empty excerpt, which ``build_report`` reports as "not
    classified from evidence" and fails on. The alternative, handing over a
    namesake's traceback, is how an assertion failure came to be filed as the
    container's ptrace fault.
    """
    leaf = node_id.split("::", 1)[1] if "::" in node_id else node_id
    path = node_id.split("::", 1)[0] if "::" in node_id else ""

    candidates = [i for i, block in enumerate(occurrences) if block["name"] == leaf]
    candidates = [i for i in candidates if i not in claimed]
    source = "banner"
    if len(candidates) > 1 and path:
        candidates = [i for i in candidates if path in occurrences[i]["text"]]
        source = "banner+file"
    if len(candidates) != 1:
        return "", "unresolved"
    claimed.add(candidates[0])
    return occurrences[candidates[0]]["text"], source


def parse_junit(path: str) -> list[dict[str, Any]]:
    """Per-test outcome, message and text from a JUnit XML report.

    Used for the required-H100 pass, because JUnit is the only one of the two
    inputs that records a skip's *reason*.
    """
    root = ET.parse(path).getroot()
    records = []
    for case in root.iter("testcase"):
        classname = case.get("classname") or ""
        node_id = f"{classname}::{case.get('name')}" if classname else str(case.get("name"))
        outcome, message, text = "passed", "", ""
        for tag, name in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            found = case.find(tag)
            if found is not None:
                outcome = name
                message = found.get("message") or ""
                text = found.text or ""
                break
        records.append(
            {
                # `classname::name`; classname is dotted
                # (`tests.unittest._torch.modules.moe.test_moe_module`), which
                # is why matching below is substring-based rather than exact.
                "node_id": node_id,
                "file": case.get("file") or "",
                "outcome": outcome,
                "message": message,
                "text": text,
            }
        )
    return records


def classify_failure(record: dict[str, Any], signatures: list[dict[str, Any]]) -> dict[str, Any]:
    """Match one failure against the registered environment signatures.

    Every clause of a signature must appear in the failure's own text. A
    failure that matches nothing is ``genuine`` --- that is the whole point:
    the classification is driven by what the failure says, so a real
    regression cannot inherit an environment verdict from its node id.
    """
    haystack = f"{record.get('message', '')}\n{record.get('text', '')}"
    for signature in signatures:
        if all(clause in haystack for clause in signature["all_of"]):
            return {
                "kind": "environment",
                "signature": signature["id"],
                "matched_clauses": list(signature["all_of"]),
            }
    return {
        "kind": "genuine",
        "signature": None,
        "matched_clauses": [],
        # Enough of the failure to act on without opening the log.
        "excerpt": haystack.strip()[-400:],
    }


def classify_skip(record: dict[str, Any], signatures: list[dict[str, Any]]) -> dict[str, Any]:
    """Match one skip's reason against the registered skip signatures.

    Same shape as :func:`classify_failure`, and for the same reason: a skip is
    only expected when its own recorded reason says why, and an unrecognised
    reason is a finding. A test that quietly starts skipping is coverage that
    disappeared, which reads exactly like a test that still passes.
    """
    reason = f"{record.get('message', '')}\n{record.get('text', '')}"
    for signature in signatures:
        if all(clause in reason for clause in signature["all_of"]):
            return {"signature": signature["id"]}
    return {"signature": None}


def required_here(node_id: str, rule: dict[str, Any]) -> bool:
    """Is this a case the Hopper gates require to actually run on this node?

    The rule is the Stage 1/2 focused Hopper selection expressed against node
    ids: any DeepSeek-V4 / SM90 case in the registered files. Exclusions are
    deliberately not a name pattern --- they are an explicit list, one entry
    per test, each carrying the evidence that this node cannot run it. A
    pattern would quietly grow to cover whatever is failing today; a list has
    to be argued for one test at a time.
    """
    lowered = node_id.lower()
    if not any(part in lowered for part in rule["node_id_any_of"]):
        return False
    return not any(entry["node_id"] == node_id for entry in rule["excluded_with_evidence"])


def build_report(
    *,
    baseline: dict[str, Any],
    registered_log: dict[str, Any],
    required_records: list[dict[str, Any]],
    required_command: str,
    device_report: dict[str, Any],
) -> dict[str, Any]:
    """Adjudicate one regression run into a pass/fail verdict plus its evidence."""
    signatures = baseline["environment_failure_signatures"]
    rule = baseline["required_h100_rule"]

    failures = []
    for record in registered_log["failures"]:
        verdict = classify_failure(record, signatures)
        failures.append(
            {
                "node_id": record["node_id"],
                "outcome": record["outcome"],
                "traceback_source": record.get("traceback_source", "banner"),
                **verdict,
            }
        )

    expected = set(baseline["expected_environment_failures"])
    observed = {f["node_id"] for f in failures}
    environment = {f["node_id"] for f in failures if f["kind"] == "environment"}
    genuine = sorted(f["node_id"] for f in failures if f["kind"] == "genuine")
    # A failure whose traceback could not be tied to it one-to-one was not
    # judged on its own evidence, whatever the signature match would have said.
    unresolved = sorted(f["node_id"] for f in failures if f["traceback_source"] == "unresolved")

    # "New" means absent from the registered baseline, reported whether or not
    # it matches a signature: a *new* environment failure is still a change.
    new_failures = sorted(observed - expected)
    resolved = sorted(expected - observed)

    required = [r for r in required_records if required_here(r["node_id"], rule)]
    skips = [
        {
            "node_id": r["node_id"],
            "reason": (r["message"] or r["text"] or "").strip()[:300],
            **classify_skip(r, baseline["expected_skip_signatures"]),
        }
        for r in required
        if r["outcome"] == "skipped"
    ]
    required_skipped = [s for s in skips if s["signature"] is None]
    # From both inputs: the focused pass spells node ids with dots and the
    # registered log with paths, so the same test can only be matched by rule,
    # not by identity. The union is what matters.
    required_failed = sorted(
        {r["node_id"] for r in required if r["outcome"] in ("failed", "error")}
        | {n for n in observed if required_here(n, rule)}
    )

    controls = baseline["protected_blackwell_controls"]
    blackwell = _blackwell_block(required_records, rule, controls, device_report)
    static = blackwell["static_dispatch_tests"]

    counts = registered_log["counts"]
    reported_failures = counts.get("failed", 0) + counts.get("error", 0)

    problems = []
    if genuine:
        problems.append(f"{len(genuine)} failure(s) match no registered environment signature")
    if unresolved:
        problems.append(
            f"{len(unresolved)} failure(s) have no traceback of their own in the log, "
            "so they were not classified from evidence"
        )
    if new_failures:
        problems.append(f"{len(new_failures)} failure(s) absent from the registered baseline")
    if required_skipped:
        problems.append(f"{len(required_skipped)} required H100 case(s) skipped")
    if required_failed:
        problems.append(f"{len(required_failed)} required H100 case(s) failed")
    if len(failures) != reported_failures:
        problems.append(
            f"the log summarised {reported_failures} failing test(s) but "
            f"{len(failures)} were recovered from it"
        )
    if not counts.get("passed"):
        problems.append("the registered run reported no passing tests")
    if not required:
        problems.append("the required-H100 pass matched no cases")
    if not controls:
        problems.append("no protected Blackwell control is registered, so none can be checked")
    for label in ("missing", "skipped", "failed"):
        if static[label]:
            problems.append(
                f"{len(static[label])} registered protected Blackwell control(s) "
                f"{label}: {static[label]}"
            )

    return {
        "evidence_label": "protected_regression_classification",
        "baseline_sha256": baseline.get("_sha256"),
        "what_gates": baseline["what_gates"],
        "registered_command": baseline["registered_command"],
        "registered_run": {
            "log": registered_log["path"],
            "exit_code": registered_log["exit_code"],
            "counts": counts,
            "failures_recovered_from_log": len(failures),
            "note": (
                "failures are classified from this run's own tracebacks, not from a "
                "re-run; the exit code is nonzero by design because the container "
                "fault below is unfixable from inside the repository"
            ),
        },
        "failures": {
            "total": len(failures),
            "environment": len(environment),
            "genuine": genuine,
            "new_against_baseline": new_failures,
            "resolved_against_baseline": resolved,
            "by_signature": {
                signature["id"]: sorted(
                    f["node_id"] for f in failures if f["signature"] == signature["id"]
                )
                for signature in signatures
            },
            "detail": sorted(failures, key=lambda f: f["node_id"]),
        },
        "required_h100_cases": {
            "rule": rule,
            "command": required_command,
            "count": len(required),
            "passed": sum(1 for r in required if r["outcome"] == "passed"),
            "failed": required_failed,
            "skipped": required_skipped,
            "skipped_with_registered_reason": {
                signature["id"]: sorted(
                    s["node_id"] for s in skips if s["signature"] == signature["id"]
                )
                for signature in baseline["expected_skip_signatures"]
            },
            "note": (
                "a skip is a finding unless its own recorded reason matches a "
                "registered skip signature; these are the cases the Hopper gates "
                "depend on, so any of them silently not running is coverage that "
                "disappeared"
            ),
        },
        "protected_blackwell_dispatch": blackwell,
        "problems": problems,
        "passed": not problems,
    }


def _blackwell_block(
    records: list[dict[str, Any]],
    rule: dict[str, Any],
    controls: list[str],
    device_report: dict[str, Any],
) -> dict[str, Any]:
    """Protected SM100 dispatch coverage, and the runtime this node cannot give.

    These tests are static: they assert which branch is selected against a
    mocked SM100 environment, so they are real evidence that the Blackwell
    *branch* is unchanged. They are not evidence that a Blackwell GPU executes
    it, and the two are recorded separately so nobody has to guess which claim
    is being made.

    Coverage is judged against a **registered inventory**, not against whatever
    the run happened to contain. Counting discovered tests could only ever say
    "the ones that ran, ran": a deleted control, a control that started
    skipping, and a run that selected no controls at all all produce a clean
    count of the empty set. The registered list turns each of those into a
    named missing control.
    """
    by_id = {r["node_id"]: r for r in records}
    present = {node_id: by_id[node_id] for node_id in controls if node_id in by_id}

    missing = sorted(set(controls) - set(present))
    skipped = sorted(n for n, r in present.items() if r["outcome"] == "skipped")
    failed = sorted(n for n, r in present.items() if r["outcome"] in ("failed", "error"))
    passed = sorted(n for n, r in present.items() if r["outcome"] == "passed")

    # Informational only: a control that exists but nobody registered is worth
    # seeing, but it is not this criterion's risk --- coverage disappearing is.
    discovered = {
        r["node_id"]
        for r in records
        if any(k in r["node_id"].lower() for k in rule["blackwell_any_of"])
    }
    devices = sorted(set(device_report.get("names", [])))
    return {
        "static_dispatch_tests": {
            "registered": len(controls),
            "count": len(present),
            "passed": len(passed),
            "passed_node_ids": passed,
            "missing": missing,
            "skipped": skipped,
            "failed": failed,
            "discovered_not_registered": sorted(discovered - set(controls)),
            "kind": "static/unit against a mocked SM100 environment on this H100 node",
        },
        "blackwell_runtime": "Not measured",
        "blackwell_runtime_reason": (
            f"no Blackwell GPU is attached to this node; the visible devices are "
            f"{devices} at compute capability {device_report.get('compute_capability')}"
        ),
    }


def load_baseline(path: str = BASELINE_PATH) -> dict[str, Any]:
    """Load the registered baseline and carry its own hash into the report."""
    import hashlib

    with open(path, "rb") as f:
        raw = f.read()
    baseline = json.loads(raw)
    baseline["_sha256"] = hashlib.sha256(raw).hexdigest()
    return baseline
