"""
Instrument Profiles

Defines the expected frequency ranges for instruments commonly used in Arabic music.
These ranges are passed to the SOTA neural pitch tracker (PENN)
to prevent octave errors — neural models can sometimes hallucinate sub-octave
harmonics if given too wide a search range.
"""

from dataclasses import dataclass


@dataclass
class InstrumentProfile:
    """
    Configuration for a specific instrument's frequency range.
    Used to constrain the pitch search space.
    """
    name: str
    display_name: str
    fmin: float
    fmax: float
    description: str


INSTRUMENTS = {
    "voice_male": InstrumentProfile(
        name="voice_male",
        display_name="Male Voice",
        fmin=100.0,
        fmax=400.0,
        description="Male singing voice (maqam style)",
    ),
    "voice_female": InstrumentProfile(
        name="voice_female",
        display_name="Female Voice",
        fmin=180.0,
        fmax=700.0,
        description="Female singing voice (maqam style)",
    ),
    "oud": InstrumentProfile(
        name="oud",
        display_name="Oud",
        fmin=80.0,
        fmax=600.0,
        description="Arabic oud (lowest string ~80Hz)",
    ),
    "nay": InstrumentProfile(
        name="nay",
        display_name="Nay",
        fmin=200.0,
        fmax=1200.0,
        description="Arabic nay flute",
    ),
    "violin": InstrumentProfile(
        name="violin",
        display_name="Violin",
        fmin=180.0,
        fmax=1000.0,
        description="Arabic violin",
    ),
    "kanun": InstrumentProfile(
        name="kanun",
        display_name="Kanun (Qanun)",
        fmin=120.0,
        fmax=1000.0,
        description="Arabic kanun (zither)",
    ),
    "general": InstrumentProfile(
        name="general",
        display_name="General / Unknown",
        fmin=180.0,
        fmax=800.0,
        description="Safe default for most Arabic melodic instruments",
    ),
}


def get_instrument_range(instrument: str) -> tuple[float, float]:
    """Get (fmin, fmax) for a given instrument. Falls back to 'general' if unknown."""
    profile = INSTRUMENTS.get(instrument.lower())
    if profile is None:
        print(f"      ⚠ Unknown instrument '{instrument}', using 'general' profile.")
        print(f"      Available: {', '.join(INSTRUMENTS.keys())}")
        profile = INSTRUMENTS["general"]
    return profile.fmin, profile.fmax


def list_instruments() -> str:
    """Return a formatted list of available instruments."""
    lines = []
    for key, p in INSTRUMENTS.items():
        lines.append(f"  {key:<15} {p.fmin:.0f}-{p.fmax:.0f} Hz   {p.description}")
    return "\n".join(lines)