"""NVIDIA profiling compatibility for GEAK v3.2."""

from .profile_adapter import profile_for_geak, profile_result_to_geak

__all__ = ["profile_for_geak", "profile_result_to_geak"]
