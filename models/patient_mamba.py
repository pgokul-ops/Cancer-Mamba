#!/usr/bin/env python3
"""
models/patient_mamba.py

Re-exports and provides dedicated interface for PatientMambaAggregator.
"""

from models.patient_aggregators import PatientMambaAggregator

PatientMamba = PatientMambaAggregator
__all__ = ["PatientMamba", "PatientMambaAggregator"]
