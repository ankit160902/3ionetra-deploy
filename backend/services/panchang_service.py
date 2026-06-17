import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any

try:
    from jyotishganit.core.astronomical import get_timescale, get_ephemeris, calculate_ayanamsa
    from jyotishganit.components.panchanga import (
        calculate_tithi, calculate_nakshatra, calculate_yoga,
        calculate_karana, calculate_vaara, solar_longitude
    )
    JYOTISH_AVAILABLE = True
except ImportError:
    JYOTISH_AVAILABLE = False
    logging.warning("jyotishganit or skyfield not available. Panchang features will be disabled.")

logger = logging.getLogger(__name__)

# Rashi index → Hindu month (Sauramana/solar system)
_RASHI_TO_MASA = [
    "Chaitra",       # 0: Mesha (0-30°)
    "Vaishakha",     # 1: Vrishabha (30-60°)
    "Jyeshtha",      # 2: Mithuna (60-90°)
    "Ashadha",       # 3: Karka (90-120°)
    "Shravana",      # 4: Simha (120-150°)
    "Bhadrapada",    # 5: Kanya (150-180°)
    "Ashwin",        # 6: Tula (180-210°)
    "Kartik",        # 7: Vrischika (210-240°)
    "Margashirsha",  # 8: Dhanu (240-270°)
    "Pausha",        # 9: Makara (270-300°)
    "Magha",         # 10: Kumbha (300-330°)
    "Phalguna",      # 11: Meena (330-360°)
]


class PanchangService:
    """
    Service for calculating Vedic Panchang (Tithi, Nakshatra, Yoga, Karana, Vaara)
    with enriched spiritual context from a static reference knowledge base.
    Uses the jyotishganit library based on Skyfield and JPL ephemeris.
    """

    def __init__(self):
        self.available = JYOTISH_AVAILABLE
        self._ts = None
        self._eph = None
        self._cached_result = None
        self._cached_at = None
        self._enriched_cached_result = None
        self._enriched_cached_at = None
        self._cache_ttl = timedelta(minutes=30)
        self._ref_data = self._load_reference_data()

    def _load_reference_data(self) -> Dict:
        """Load panchang_reference.json static knowledge base."""
        ref_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "panchang_reference.json"
        )
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded panchang reference data ({len(data)} sections)")
            return data
        except Exception as e:
            logger.warning(f"Could not load panchang reference data: {e}")
            return {}

    def _ensure_data(self):
        """Ensure astronomical data is loaded."""
        if not self.available:
            return False
        try:
            if self._ts is None:
                try:
                    self._ts = get_timescale()
                except Exception as e:
                    # VPC egress blocks finals2000A.all download — use builtin leap-second table
                    logger.warning(f"Timescale download failed ({e}), falling back to builtin")
                    from skyfield.api import load as _sf_load
                    self._ts = _sf_load.timescale(builtin=True)
            if self._eph is None:
                self._eph = get_ephemeris()
            return True
        except Exception as e:
            logger.error(f"Failed to load astronomical data: {e}")
            self.available = False
            return False

    def _is_cache_valid(self, cached_result: Dict, cached_at: datetime, now: datetime) -> bool:
        """Check if a cached result is still valid (within TTL and same date)."""
        if not cached_result or not cached_at:
            return False
        if now - cached_at >= self._cache_ttl:
            return False
        # Invalidate if the date has changed (midnight boundary)
        if cached_result.get("date") != now.strftime("%Y-%m-%d"):
            return False
        return True

    def get_panchang(self, dt: datetime, latitude: float = 28.6139, longitude: float = 77.2090, timezone_offset: float = 5.5) -> Dict[str, Any]:
        """
        Calculate Panchang for a given datetime and location.
        Defaults to New Delhi (28.6139, 77.2090) and IST (+5.5).
        """
        using_defaults = (latitude == 28.6139 and longitude == 77.2090 and timezone_offset == 5.5)
        now = datetime.now()
        if using_defaults and self._is_cache_valid(self._cached_result, self._cached_at, now):
            return self._cached_result

        if not self._ensure_data():
            return {"error": "Panchang service unavailable"}

        try:
            # Convert to UTC Skyfield time for ayanamsa
            utc_dt = dt - timedelta(hours=timezone_offset)
            t = self._ts.utc(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour, utc_dt.minute, utc_dt.second)

            # Calculate Ayanamsa (True Chitra Paksha / Lahiri)
            ayanamsa = calculate_ayanamsa(t)

            # Calculate the five limbs
            tithi = calculate_tithi(dt, timezone_offset)
            nakshatra = calculate_nakshatra(dt, timezone_offset, ayanamsa)
            yoga = calculate_yoga(dt, timezone_offset, ayanamsa)
            karana = calculate_karana(dt, timezone_offset)
            vaara = calculate_vaara(dt)

            result = {
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "tithi": tithi,
                "nakshatra": nakshatra,
                "yoga": yoga,
                "karana": karana,
                "vaara": vaara,
                "ayanamsa": round(ayanamsa, 4),
                "location": {"lat": latitude, "lon": longitude}
            }
            if using_defaults:
                self._cached_result = result
                self._cached_at = now
            return result
        except Exception as e:
            logger.exception(f"Error calculating panchang: {e}")
            return {"error": str(e)}

    # ── Enrichment helpers ──────────────────────────────────────────────

    def _get_paksha(self, tithi_name: str) -> str:
        """Extract paksha from the tithi name (already includes Shukla/Krishna prefix)."""
        if tithi_name.startswith("Shukla") or tithi_name == "Purnima":
            return "Shukla"
        elif tithi_name.startswith("Krishna") or tithi_name == "Amavasya":
            return "Krishna"
        return ""

    def _get_base_tithi(self, tithi_name: str) -> str:
        """Extract base tithi name without paksha prefix."""
        if tithi_name in ("Purnima", "Amavasya"):
            return tithi_name
        for prefix in ("Shukla ", "Krishna "):
            if tithi_name.startswith(prefix):
                return tithi_name[len(prefix):]
        return tithi_name

    def _calculate_masa(self, dt: datetime, timezone_offset: float, ayanamsa: float) -> str:
        """
        Compute Hindu month (masa) from sidereal solar longitude.
        Uses the Sauramana (solar) system. Accepts pre-computed ayanamsa
        to avoid redundant astronomical calculations.
        """
        if not self._ensure_data():
            return ""
        try:
            utc_dt = dt - timedelta(hours=timezone_offset)
            t = self._ts.utc(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour, utc_dt.minute, utc_dt.second)
            sun_lon = solar_longitude(t)
            sidereal = (sun_lon - ayanamsa) % 360
            rashi_index = int(sidereal / 30)
            return _RASHI_TO_MASA[rashi_index]
        except Exception as e:
            logger.warning(f"Error calculating masa: {e}")
            return ""

    # ── Festival detection ──────────────────────────────────────────────

    def _lookup_festival_by_date(self, dt: datetime) -> str:
        """
        Primary festival detection: pre-computed Gregorian dates (2025-2030).
        Cross-verified against DrikPanchang, CalendarLabs, TimeAndDate.
        """
        date_key = dt.strftime("%Y-%m-%d")
        return self._ref_data.get("festival_dates", {}).get(date_key, "")

    def _lookup_festival_by_tithi_masa(self, masa: str, paksha: str, tithi_name: str) -> str:
        """
        Fallback festival detection: tithi + masa combination.
        Used for dates outside the pre-computed 2025-2030 range.
        Note: solar masa may differ from lunar masa by ~15 days at month
        boundaries, so this is best-effort for dates beyond 2030.
        """
        festivals = self._ref_data.get("festivals_by_tithi_masa", {})
        base_tithi = self._get_base_tithi(tithi_name)

        # Try paksha-specific key: {Masa}_{Paksha}_{BaseTithi}
        key1 = f"{masa}_{paksha}_{base_tithi}"
        if key1 in festivals:
            return festivals[key1]

        # Try paksha-independent key for Purnima/Amavasya
        if base_tithi in ("Purnima", "Amavasya"):
            key2 = f"{masa}_{base_tithi}"
            if key2 in festivals:
                return festivals[key2]

        return ""

    def _detect_festival(self, dt: datetime, masa: str, paksha: str, tithi_name: str) -> str:
        """
        Detect festival: date-based lookup first (accurate for 2025-2030),
        then tithi+masa fallback for dates outside that range.
        """
        # Primary: pre-computed date lookup
        festival = self._lookup_festival_by_date(dt)
        if festival:
            return festival

        # Fallback: tithi+masa (best-effort for dates outside 2025-2030)
        year = dt.year
        if year < 2025 or year > 2030:
            return self._lookup_festival_by_tithi_masa(masa, paksha, tithi_name)

        return ""

    # ── Festival context lookup ───────────────────────────────────────

    def _get_festival_context(self, festival_name: str) -> Dict:
        """
        Look up spiritual context (mantra, rituals, significance) for a festival.
        Uses longest-substring matching to handle composite names like
        'Naraka Chaturdashi / Chhoti Diwali' → matches 'naraka chaturdashi' over 'diwali'.
        """
        if not festival_name:
            return {}
        contexts = self._ref_data.get("festival_context", {})
        name_lower = festival_name.lower()
        best_match = None
        best_len = 0
        for key in contexts:
            if key in name_lower and len(key) > best_len:
                best_match = key
                best_len = len(key)
        return contexts.get(best_match, {}) if best_match else {}

    # ── Upcoming festival lookahead ────────────────────────────────────

    def _get_upcoming_festivals(self, dt: datetime, lookahead_days: int = 3) -> list:
        """
        Find festivals in the next N days from the pre-computed festival_dates.
        Returns list of {date, days_away, festival} dicts, sorted by date.
        """
        festival_dates = self._ref_data.get("festival_dates", {})
        upcoming = []
        for i in range(1, lookahead_days + 1):
            future = dt + timedelta(days=i)
            key = future.strftime("%Y-%m-%d")
            if key in festival_dates:
                fest_name = festival_dates[key]
                ctx = self._get_festival_context(fest_name)
                entry = {
                    "date": key,
                    "days_away": i,
                    "festival": fest_name,
                }
                if ctx.get("mantra"):
                    entry["mantra"] = ctx["mantra"]
                upcoming.append(entry)
        return upcoming

    # ── Main enrichment method ──────────────────────────────────────────

    def get_enriched_panchang(self, dt: datetime, latitude: float = 28.6139, longitude: float = 77.2090, timezone_offset: float = 5.5) -> Dict[str, Any]:
        """
        Get panchang with enriched spiritual context from reference data.
        Adds: paksha, masa, tithi significance, nakshatra quality, yoga quality, festival detection.
        """
        using_defaults = (latitude == 28.6139 and longitude == 77.2090 and timezone_offset == 5.5)
        now = datetime.now()
        if using_defaults and self._is_cache_valid(self._enriched_cached_result, self._enriched_cached_at, now):
            return self._enriched_cached_result

        # Start with base panchang
        result = self.get_panchang(dt, latitude, longitude, timezone_offset)
        if "error" in result:
            return result

        tithi_name = result.get("tithi", "")
        nakshatra_name = result.get("nakshatra", "")
        yoga_name = result.get("yoga", "")

        # Paksha from tithi name
        paksha = self._get_paksha(tithi_name)
        base_tithi = self._get_base_tithi(tithi_name)
        result["paksha"] = paksha

        paksha_data = self._ref_data.get("paksha_info", {}).get(paksha, {})
        result["paksha_significance"] = paksha_data.get("significance", "")

        # Masa — reuse ayanamsa from base panchang (avoids redundant computation)
        masa = self._calculate_masa(dt, timezone_offset, result["ayanamsa"])
        result["masa"] = masa

        masa_data = self._ref_data.get("masa_info", {}).get(masa, {})
        result["masa_significance"] = masa_data.get("significance", "")
        result["masa_season"] = masa_data.get("season", "")

        # Tithi significance
        tithi_data = self._ref_data.get("tithi_significance", {}).get(base_tithi, {})
        result["tithi_significance"] = tithi_data.get("significance", "")
        result["tithi_deity"] = tithi_data.get("deity", "")
        result["tithi_good_for"] = tithi_data.get("good_for", "")
        result["tithi_vrat"] = tithi_data.get("vrat", "")

        # Nakshatra significance
        nak_data = self._ref_data.get("nakshatra_significance", {}).get(nakshatra_name, {})
        result["nakshatra_quality"] = nak_data.get("quality", "")
        result["nakshatra_deity"] = nak_data.get("deity", "")
        result["nakshatra_good_for"] = nak_data.get("good_for", "")

        # Yoga significance
        yoga_data = self._ref_data.get("yoga_significance", {}).get(yoga_name, {})
        result["yoga_quality"] = yoga_data.get("quality", "")
        result["yoga_nature"] = yoga_data.get("nature", "")

        # Festival detection (date-based primary, tithi+masa fallback)
        festival = self._detect_festival(dt, masa, paksha, tithi_name)
        result["festival"] = festival

        # Festival spiritual context (mantra, rituals, significance)
        fest_ctx = self._get_festival_context(festival)
        result["festival_mantra"] = fest_ctx.get("mantra", "")
        result["festival_rituals"] = fest_ctx.get("rituals", "")
        result["festival_significance"] = fest_ctx.get("significance", "")
        result["festival_deity"] = fest_ctx.get("deity", "")

        # Enhanced special day info
        result["special_day"] = self._get_enhanced_special_day(tithi_name, festival)

        # Upcoming festivals (next 3 days lookahead)
        result["upcoming_festivals"] = self._get_upcoming_festivals(dt)

        if using_defaults:
            self._enriched_cached_result = result
            self._enriched_cached_at = now

        return result

    def _get_enhanced_special_day(self, tithi_name: str, festival: str) -> str:
        """Build an enhanced special day string combining tithi type and festival."""
        parts = []
        tithi_lower = tithi_name.lower()

        if "ekadashi" in tithi_lower:
            parts.append("Today is an auspicious Ekadashi — a day for fasting, prayer, and spiritual purification dedicated to Lord Vishnu.")
        elif tithi_name == "Amavasya":
            parts.append("Today is Amavasya (New Moon) — a powerful time for ancestral prayers (Pitru Tarpan) and inner reflection.")
        elif tithi_name == "Purnima":
            parts.append("Today is Purnima (Full Moon) — a time of fulfillment, Satyanarayan Puja, and spiritual abundance.")
        elif "chaturdashi" in tithi_lower:
            parts.append("Today is Chaturdashi — associated with Lord Shiva. Observe Pradosh or Shivaratri Vrat if drawn.")

        if festival:
            parts.append(f"Festival: {festival}.")

        return " ".join(parts)

    def get_special_day_info(self, panchang: Dict[str, Any]) -> str:
        """
        Identify if today is a special spiritual day (Ekadashi, Amavasya, etc.)
        Backward-compatible method — prefer get_enriched_panchang() for richer data.
        """
        tithi = panchang.get("tithi", "").lower()
        if "ekadashi" in tithi:
            return "Today is an auspicious Ekadashi. It is a day for fasting, prayer, and spiritual purification."
        elif "amavasya" in tithi:
            return "Today is Amavasya (New Moon). A powerful time for ancestral prayers and inner reflection."
        elif "purnima" in tithi:
            return "Today is Purnima (Full Moon). A time of fulfillment and spiritual abundance."
        return ""

_panchang_service = None

def get_panchang_service() -> PanchangService:
    global _panchang_service
    if _panchang_service is None:
        _panchang_service = PanchangService()
    return _panchang_service
