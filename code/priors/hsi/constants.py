"""Non-sealed constants shared by the HSI runtime and its tests."""

# Keep this value local to the HSI guidance layer.  The sealed metrics module
# owns the reporting copy; a test asserts that the two contracts stay equal.
FLOOR_EXCLUSION_HEIGHT_M = 0.02
SDF_MARGIN_M = 0.0

__all__ = ["FLOOR_EXCLUSION_HEIGHT_M", "SDF_MARGIN_M"]
