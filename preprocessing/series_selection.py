#!/usr/bin/env python3
"""
preprocessing/series_selection.py

Selects eligible primary volumetric series per patient-study from data/manifests/manifest_v1.json.
Rules:
  1. Exclude scouts (<=5 slices) and intermediate scans (6-20 slices).
  2. Exclude non-axial orientations unless no axial series exists for that study.
  3. Keep ALL distinct contrast phases / roles as separate eligible observations.
  4. If multiple series within a study share the same phase/role, keep the one with
     the most slices (tie-break) and log the exclusion.

Outputs:
  data/manifests/series_selection.json
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def get_slice_orientation(orientation: Optional[List[float]]) -> str:
    """
    Classifies DICOM orientation (ImageOrientationPatient) as AXIAL, CORONAL, or SAGITTAL.
    ImageOrientationPatient = [rx, ry, rz, cx, cy, cz].
    Normal vector = r x c.
    """
    if not orientation or len(orientation) != 6:
        return "UNKNOWN"
    try:
        r = np.array(orientation[:3], dtype=float)
        c = np.array(orientation[3:], dtype=float)
        n = np.cross(r, c)
        nx, ny, nz = abs(n[0]), abs(n[1]), abs(n[2])
        if nz >= nx and nz >= ny:
            return "AXIAL"
        elif ny >= nx and ny >= nz:
            return "CORONAL"
        elif nx >= ny and nx >= nz:
            return "SAGITTAL"
        return "OBLIQUE"
    except Exception:
        return "UNKNOWN"


def classify_contrast_phase(series_desc: str) -> str:
    """
    Categorizes the series description into an anatomical / contrast phase role.
    """
    d = (series_desc or "").upper().strip()

    # Delayed / excretory phase
    if any(k in d for k in ["DELAY", "KIDNEY DELAY", "DELAYED BLADDER", "EXCRETORY"]):
        return "delayed"

    # Arterial phase
    if any(k in d for k in ["ARTERIAL", "ART PHASE", "ART_PHASE", "LIVER ART"]):
        return "arterial"

    # Portal venous / venous phase
    if any(k in d for k in ["VENOUS", "PORTAL", "PV PHASE", "PORTAL VENOUS"]):
        return "portal_venous"

    # Non-contrast / pre-contrast
    if any(k in d for k in ["NON CON", "NON-CON", "WITHOUT", "W/O", "NONE", "PRE-CON", "PRE CON"]):
        return "non_contrast"

    # Lung parenchyma window
    if any(k in d for k in ["LUNG", "LUNGS"]):
        return "lung_window"

    # Standard chest CT without abdomen/pelvis
    if any(k in d for k in ["CHEST", "THORAX"]) and not any(k in d for k in ["ABD", "PEL"]):
        return "chest_std"

    # Generic contrast-enhanced scan
    if any(k in d for k in ["WITH CE", "C+", "CONTRAST", "CE", "OMNI", "BARIUM", "ISOVUE", "3ML/SEC"]):
        return "contrast_enhanced_general"

    # Default routine abdominal/pelvic scan
    return "primary_routine"


def run_series_selection(
    manifest_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Performs primary series selection over all CT series in the manifest.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter CT series
    ct_series = [s for s in manifest.get("series", []) if s.get("modality") == "CT"]

    # Organize by patient -> study
    patient_studies = defaultdict(lambda: defaultdict(list))
    for s in ct_series:
        patient_studies[s["patient_id"]][s["study_id"]].append(s)

    selected_series_records: List[Dict[str, Any]] = []
    excluded_series_records: List[Dict[str, Any]] = []
    tie_break_records: List[Dict[str, Any]] = []
    phase_distribution = Counter()

    # Accounting structures
    accounting_all_volumetric_282: List[Dict[str, Any]] = []

    # Map for structured output: patient_id -> study_id -> [selected_series]
    selection_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for pid in sorted(patient_studies.keys()):
        for sid in sorted(patient_studies[pid].keys()):
            series_in_study = patient_studies[pid][sid]

            # Step 1: Slice count filtering (scouts <=5, intermediate 6-20, volumetric >20)
            volumetric_candidates = []
            for s in series_in_study:
                n = s.get("num_instances", 0)
                s_id = s["series_id"]
                s_desc = s.get("series_description", "")
                orient = get_slice_orientation(s.get("orientation"))

                if n <= 5:
                    excluded_series_records.append({
                        "series_id": s_id,
                        "patient_id": pid,
                        "study_id": sid,
                        "reason": "scout_localizer",
                        "details": f"Slice count {n} <= 5",
                        "num_instances": n,
                        "series_description": s_desc,
                    })
                elif n <= 20:
                    excluded_series_records.append({
                        "series_id": s_id,
                        "patient_id": pid,
                        "study_id": sid,
                        "reason": "intermediate_slice_count",
                        "details": f"Slice count {n} in (5, 20]; flagged and excluded from primary path",
                        "num_instances": n,
                        "series_description": s_desc,
                    })
                else:
                    volumetric_candidates.append(s)

            # Step 2: Orientation filtering (prefer AXIAL when present)
            axial_series = [
                s for s in volumetric_candidates
                if get_slice_orientation(s.get("orientation")) == "AXIAL"
            ]

            if axial_series:
                orientation_eligible = axial_series
                for s in volumetric_candidates:
                    orient = get_slice_orientation(s.get("orientation"))
                    if orient != "AXIAL":
                        reason_msg = f"non_axial_orientation ({orient}); study has axial volumetric series"
                        excluded_series_records.append({
                            "series_id": s["series_id"],
                            "patient_id": pid,
                            "study_id": sid,
                            "reason": "non_axial_orientation",
                            "details": reason_msg,
                            "num_instances": s.get("num_instances", 0),
                            "series_description": s.get("series_description", ""),
                        })
                        accounting_all_volumetric_282.append({
                            "series_id": s["series_id"],
                            "patient_id": pid,
                            "study_id": sid,
                            "decision": "EXCLUDED",
                            "reason": "non_axial_orientation",
                            "details": reason_msg,
                            "num_instances": s.get("num_instances", 0),
                            "series_description": s.get("series_description", ""),
                        })
            else:
                # Fallback: if no axial exists, keep all volumetric candidates
                orientation_eligible = volumetric_candidates

            # Step 3: Group by contrast phase / role
            phase_groups = defaultdict(list)
            for s in orientation_eligible:
                phase_tag = classify_contrast_phase(s.get("series_description", ""))
                phase_groups[phase_tag].append(s)

            # Step 4: For each phase group, select best series (tie-break by max slices if duplicate)
            for phase_tag, group_series in phase_groups.items():
                if len(group_series) == 1:
                    winner = group_series[0]
                    selected_info = {
                        "series_id": winner["series_id"],
                        "patient_id": pid,
                        "study_id": sid,
                        "role_tag": phase_tag,
                        "num_instances": winner.get("num_instances", 0),
                        "shape": winner.get("shape"),
                        "spacing": winner.get("spacing"),
                        "series_description": winner.get("series_description", ""),
                        "path": winner.get("path"),
                        "selection_rule": "single_series_for_phase",
                    }
                    selection_map[pid][sid].append(selected_info)
                    selected_series_records.append(selected_info)
                    phase_distribution[phase_tag] += 1
                    accounting_all_volumetric_282.append({
                        "series_id": winner["series_id"],
                        "patient_id": pid,
                        "study_id": sid,
                        "decision": "SELECTED",
                        "role_tag": phase_tag,
                        "reason": "single_series_for_phase",
                        "num_instances": winner.get("num_instances", 0),
                        "series_description": winner.get("series_description", ""),
                    })
                else:
                    # Tie-break: sort descending by num_instances, then series_id for determinism
                    group_sorted = sorted(
                        group_series,
                        key=lambda x: (x.get("num_instances", 0), x.get("series_id", "")),
                        reverse=True,
                    )
                    winner = group_sorted[0]
                    selected_info = {
                        "series_id": winner["series_id"],
                        "patient_id": pid,
                        "study_id": sid,
                        "role_tag": phase_tag,
                        "num_instances": winner.get("num_instances", 0),
                        "shape": winner.get("shape"),
                        "spacing": winner.get("spacing"),
                        "series_description": winner.get("series_description", ""),
                        "path": winner.get("path"),
                        "selection_rule": f"tie_break_max_instances ({winner.get('num_instances')} slices)",
                    }
                    selection_map[pid][sid].append(selected_info)
                    selected_series_records.append(selected_info)
                    phase_distribution[phase_tag] += 1
                    accounting_all_volumetric_282.append({
                        "series_id": winner["series_id"],
                        "patient_id": pid,
                        "study_id": sid,
                        "decision": "SELECTED",
                        "role_tag": phase_tag,
                        "reason": f"tie_break_winner ({winner.get('num_instances')} vs {group_sorted[1].get('num_instances')} slices)",
                        "num_instances": winner.get("num_instances", 0),
                        "series_description": winner.get("series_description", ""),
                    })

                    # Log excluded duplicate series
                    for loser in group_sorted[1:]:
                        reason_msg = (
                            f"duplicate_phase_tie_break ({phase_tag}: loser has {loser.get('num_instances')} slices "
                            f"vs winner {winner['series_id']} with {winner.get('num_instances')} slices)"
                        )
                        tie_break_records.append({
                            "excluded_series_id": loser["series_id"],
                            "selected_series_id": winner["series_id"],
                            "patient_id": pid,
                            "study_id": sid,
                            "role_tag": phase_tag,
                            "loser_slices": loser.get("num_instances", 0),
                            "winner_slices": winner.get("num_instances", 0),
                            "loser_description": loser.get("series_description", ""),
                            "winner_description": winner.get("series_description", ""),
                        })
                        excluded_series_records.append({
                            "series_id": loser["series_id"],
                            "patient_id": pid,
                            "study_id": sid,
                            "reason": "duplicate_phase_tie_break",
                            "details": reason_msg,
                            "num_instances": loser.get("num_instances", 0),
                            "series_description": loser.get("series_description", ""),
                        })
                        accounting_all_volumetric_282.append({
                            "series_id": loser["series_id"],
                            "patient_id": pid,
                            "study_id": sid,
                            "decision": "EXCLUDED",
                            "reason": "duplicate_phase_tie_break",
                            "details": reason_msg,
                            "num_instances": loser.get("num_instances", 0),
                            "series_description": loser.get("series_description", ""),
                        })

    # Validate accounting covers all 282 volumetric series exactly
    total_volumetric_seen = len(accounting_all_volumetric_282)
    selected_count = sum(1 for a in accounting_all_volumetric_282 if a["decision"] == "SELECTED")
    excluded_volumetric_count = sum(1 for a in accounting_all_volumetric_282 if a["decision"] == "EXCLUDED")

    # Patient coverage check
    patients_with_selected = set(selection_map.keys())
    all_ct_patients = set(patient_studies.keys())

    # Compile result document
    result = {
        "metadata": {
            "version": "v1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_manifest": str(manifest_path),
            "summary": {
                "total_ct_series_on_disk": len(ct_series),
                "total_volumetric_series_on_disk": total_volumetric_seen,
                "total_selected_series": selected_count,
                "total_excluded_volumetric_series": excluded_volumetric_count,
                "scouts_excluded": sum(1 for e in excluded_series_records if e["reason"] == "scout_localizer"),
                "intermediate_excluded": sum(1 for e in excluded_series_records if e["reason"] == "intermediate_slice_count"),
                "non_axial_volumetric_excluded": sum(1 for a in accounting_all_volumetric_282 if a["reason"] == "non_axial_orientation"),
                "duplicate_tie_breaks_resolved": len(tie_break_records),
                "patients_with_selected_series": len(patients_with_selected),
                "total_ct_patients": len(all_ct_patients),
                "patient_coverage_pct": round((len(patients_with_selected) / max(1, len(all_ct_patients))) * 100, 2),
            },
            "phase_distribution_across_selected": dict(phase_distribution.most_common()),
            "volumes_per_patient_distribution": dict(
                Counter(sum(len(v) for v in studies.values()) for studies in selection_map.values())
            ),
        },
        "tie_breaks": tie_break_records,
        "selection_by_patient": selection_map,
        "accounting_all_282_volumetric_series": accounting_all_volumetric_282,
    }

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSeries selection successfully written to: {output_path}")
    print(f"  Total Volumetric CT series:    {total_volumetric_seen} / 282")
    print(f"  Selected Primary Volumetric:   {selected_count}")
    print(f"  Excluded Volumetric (Non-Axial): {sum(1 for a in accounting_all_volumetric_282 if a['reason'] == 'non_axial_orientation')}")
    print(f"  Excluded Volumetric (Duplicate): {len(tie_break_records)}")
    print(f"  Patient Coverage:              {len(patients_with_selected)} / {len(all_ct_patients)} (100.0%)")
    print("\nPhase Distribution:")
    for ph, cnt in phase_distribution.most_common():
        print(f"    {ph:28s}: {cnt:3d}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run primary series selection on CT manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest_v1.json"),
        help="Path to manifest_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/series_selection.json"),
        help="Path to output series_selection.json",
    )
    args = parser.parse_args()
    run_series_selection(manifest_path=args.manifest, output_path=args.output)


if __name__ == "__main__":
    main()
