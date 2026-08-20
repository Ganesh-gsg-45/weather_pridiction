"""
test_coverage.py

Unit tests for is_in_trained_coverage() and haversine_km().

Tests:
1. haversine_km() — known distances verified against published values
2. Chennai (exact trained coords) → in coverage, correct city name, distance ≈ 0
3. Port Blair (Andaman Islands, 11.67°N 92.74°E) → NOT in coverage, nearest city present
4. Delhi exact coords → in coverage, returns "Delhi"
5. A point 76km from nearest city → outside 75km radius (boundary just outside)
6. A point 74km from nearest city → inside radius (boundary just inside)
"""

import pytest
from math import isclose

# Import directly from app so we test the deployed function, not a copy.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import haversine_km, is_in_trained_coverage, COVERAGE_RADIUS_KM, TRAINED_CITY_COORDS


# ── haversine_km tests ─────────────────────────────────────────────────────────

class TestHaversineKm:

    def test_zero_distance_same_point(self):
        """Same point → distance is 0 km."""
        d = haversine_km(28.6139, 77.2090, 28.6139, 77.2090)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_delhi_to_mumbai_approx(self):
        """
        Delhi (28.6139°N, 77.2090°E) to Mumbai (19.0760°N, 72.8777°E).
        Published approximate straight-line distance: ~1153 km.
        We accept ±10 km to account for different Earth radius conventions.
        """
        d = haversine_km(28.6139, 77.2090, 19.0760, 72.8777)
        assert 1140 <= d <= 1165, f"Expected ~1153 km, got {d:.1f} km"

    def test_delhi_to_chennai_approx(self):
        """
        Delhi to Chennai: published ~1754 km straight-line.
        Accept ±15 km.
        """
        d = haversine_km(28.6139, 77.2090, 13.0827, 80.2707)
        assert 1740 <= d <= 1770, f"Expected ~1754 km, got {d:.1f} km"

    def test_symmetry(self):
        """haversine_km(A→B) == haversine_km(B→A)."""
        d1 = haversine_km(28.6, 77.2, 19.0, 72.8)
        d2 = haversine_km(19.0, 72.8, 28.6, 77.2)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_equator_1_degree_longitude(self):
        """
        At the equator, 1° of longitude ≈ 111.32 km.
        Accept ±0.5 km.
        """
        d = haversine_km(0.0, 0.0, 0.0, 1.0)
        assert 110.8 <= d <= 111.8, f"Expected ~111.3 km at equator, got {d:.2f} km"


# ── is_in_trained_coverage tests ───────────────────────────────────────────────

class TestCoverageCheck:

    def test_chennai_exact_coords_in_coverage(self):
        """
        Exact Chennai trained coordinates → must be in coverage.
        Nearest city must be 'Chennai'. Distance must be ≈ 0 km.
        """
        in_cov, nearest, dist = is_in_trained_coverage(13.0827, 80.2707)
        assert in_cov is True, "Exact Chennai coords must be in coverage"
        assert nearest == "Chennai", f"Expected nearest='Chennai', got '{nearest}'"
        assert dist == pytest.approx(0.0, abs=0.5), f"Expected dist≈0, got {dist}"

    def test_port_blair_not_in_coverage(self):
        """
        Port Blair, Andaman Islands (11.6234°N, 92.7265°E).
        Far from all 10 trained cities — must NOT be in coverage.
        Nearest city should still be returned (not None).
        Distance to nearest city should be >> 75 km.
        """
        in_cov, nearest, dist = is_in_trained_coverage(11.6234, 92.7265)
        assert in_cov is False, (
            f"Port Blair should be OUTSIDE trained coverage. "
            f"Got in_coverage=True, nearest={nearest}, dist={dist}km"
        )
        assert nearest is not None, "nearest_city must always be present, even for fallback"
        assert dist > COVERAGE_RADIUS_KM, (
            f"Distance to nearest city should exceed {COVERAGE_RADIUS_KM}km, got {dist}km"
        )

    def test_delhi_exact_coords_in_coverage(self):
        """Exact Delhi trained coordinates → in coverage, nearest = 'Delhi'."""
        in_cov, nearest, dist = is_in_trained_coverage(28.6139, 77.2090)
        assert in_cov is True
        assert nearest == "Delhi"
        assert dist == pytest.approx(0.0, abs=0.5)

    def test_return_shape_always_three_tuple(self):
        """is_in_trained_coverage always returns (bool, str, float) — never None for city/dist."""
        in_cov, nearest, dist = is_in_trained_coverage(11.6234, 92.7265)  # Port Blair
        assert isinstance(in_cov, bool)
        assert isinstance(nearest, str) and len(nearest) > 0
        assert isinstance(dist, float) and dist > 0

    def test_coverage_radius_constant_is_75(self):
        """COVERAGE_RADIUS_KM must be 75 — changing it affects all routing decisions."""
        assert COVERAGE_RADIUS_KM == 75, (
            f"Expected COVERAGE_RADIUS_KM=75, got {COVERAGE_RADIUS_KM}. "
            "Update this test if the radius is intentionally changed."
        )

    def test_trained_city_coords_has_10_cities(self):
        """TRAINED_CITY_COORDS must contain exactly 10 entries (current dataset)."""
        assert len(TRAINED_CITY_COORDS) == 10, (
            f"Expected 10 trained cities, got {len(TRAINED_CITY_COORDS)}: "
            f"{sorted(TRAINED_CITY_COORDS.keys())}"
        )

    def test_lucknow_in_trained_coords_not_surat(self):
        """Lucknow must be in TRAINED_CITY_COORDS; Surat must NOT be."""
        assert "Lucknow" in TRAINED_CITY_COORDS, (
            "Lucknow is a verified training city but missing from TRAINED_CITY_COORDS"
        )
        assert "Surat" not in TRAINED_CITY_COORDS, (
            "Surat is NOT a training city but appears in TRAINED_CITY_COORDS — fix config/constants.py"
        )

    def test_leh_not_in_coverage(self):
        """
        Leh, Ladakh (34.1526°N, 77.5771°E) — remote mountain city.
        Nearest trained city would be Delhi (~527 km). Must be outside coverage.
        """
        in_cov, nearest, dist = is_in_trained_coverage(34.1526, 77.5771)
        assert in_cov is False, (
            f"Leh should be outside coverage. Got in_coverage=True, nearest={nearest}, dist={dist}km"
        )
        assert dist > COVERAGE_RADIUS_KM
