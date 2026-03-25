"""
Build gold standard JSON files with canonical bases.
=====================================================
Source: Wikidata, CIA World Factbook, IUCN Red List (verified 2024)
Run: python gold_standards/build.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fca_engine import FormalContext, full_exploration
from domain import COUNTRIES_DOMAIN, ANIMALS_DOMAIN, COUNTRIES_30


# ── Mock Oracle for canonical basis computation ──────────────────────────────

class GoldOracle:
    """Perfect oracle using complete gold standard."""
    def __init__(self, gold: dict[str, set[str]]):
        self.gold = gold

    def confirm_implication(self, premise, conclusion, context):
        for name in sorted(self.gold):
            if name not in context.objects:
                attrs = self.gold[name]
                if premise <= attrs and not conclusion <= attrs:
                    return (False, name, attrs)
        return (True, None, None)


# ═════════════════════════════════════════════════════════════════════════════
# D1: COUNTRIES — 50 countries × 15 attributes
# ═════════════════════════════════════════════════════════════════════════════
# Attributes: is_in_europe, is_in_asia, has_coastline, is_island_nation,
#   is_UN_member, is_NATO_member, is_EU_member, is_G7,
#   has_nuclear_weapons, population_over_50M, population_over_100M,
#   gdp_top_20, is_democracy, is_monarchy, has_official_english

COUNTRIES_GOLD: dict[str, set[str]] = {
    # ── Europe ───────────────────────────────────────────────────────────
    "united_kingdom": {
        "is_in_europe", "has_coastline", "is_island_nation",
        "is_UN_member", "is_NATO_member", "is_G7",
        "has_nuclear_weapons", "population_over_50M",
        "gdp_top_20", "is_democracy", "is_monarchy", "has_official_english",
    },
    "france": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member", "is_G7",
        "has_nuclear_weapons", "population_over_50M",
        "gdp_top_20", "is_democracy",
    },
    "germany": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member", "is_G7",
        "population_over_50M",
        "gdp_top_20", "is_democracy",
    },
    "italy": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member", "is_G7",
        "population_over_50M",
        "gdp_top_20", "is_democracy",
    },
    "spain": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "gdp_top_20", "is_democracy", "is_monarchy",
    },
    "netherlands": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "gdp_top_20", "is_democracy", "is_monarchy",
    },
    "poland": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "is_democracy",
    },
    "sweden": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "is_democracy", "is_monarchy",
    },
    "norway": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member",
        "is_democracy", "is_monarchy",
    },
    "switzerland": {
        "is_in_europe",
        "is_UN_member",
        "gdp_top_20", "is_democracy",
    },
    "greece": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "is_democracy",
    },
    "portugal": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "is_democracy",
    },
    "belgium": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "is_democracy", "is_monarchy",
    },
    "austria": {
        "is_in_europe",
        "is_UN_member", "is_EU_member",
        "is_democracy",
    },
    "finland": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "is_democracy",
    },
    "ireland": {
        "is_in_europe", "has_coastline", "is_island_nation",
        "is_UN_member", "is_EU_member",
        "is_democracy", "has_official_english",
    },
    "denmark": {
        "is_in_europe", "has_coastline",
        "is_UN_member", "is_NATO_member", "is_EU_member",
        "is_democracy", "is_monarchy",
    },

    # ── Asia ─────────────────────────────────────────────────────────────
    "china": {
        "is_in_asia", "has_coastline",
        "is_UN_member",
        "has_nuclear_weapons", "population_over_50M", "population_over_100M",
        "gdp_top_20",
    },
    "japan": {
        "is_in_asia", "has_coastline", "is_island_nation",
        "is_UN_member", "is_G7",
        "population_over_50M", "population_over_100M",
        "gdp_top_20", "is_democracy", "is_monarchy",
    },
    "south_korea": {
        "is_in_asia", "has_coastline",
        "is_UN_member",
        "population_over_50M",
        "gdp_top_20", "is_democracy",
    },
    "india": {
        "is_in_asia", "has_coastline",
        "is_UN_member",
        "has_nuclear_weapons", "population_over_50M", "population_over_100M",
        "gdp_top_20", "is_democracy", "has_official_english",
    },
    "indonesia": {
        "is_in_asia", "has_coastline", "is_island_nation",
        "is_UN_member",
        "population_over_50M", "population_over_100M",
        "gdp_top_20", "is_democracy",
    },
    "thailand": {
        "is_in_asia", "has_coastline",
        "is_UN_member",
        "population_over_50M",
        "is_democracy", "is_monarchy",
    },
    "philippines": {
        "is_in_asia", "has_coastline", "is_island_nation",
        "is_UN_member",
        "population_over_50M", "population_over_100M",
        "is_democracy", "has_official_english",
    },
    "pakistan": {
        "is_in_asia", "has_coastline",
        "is_UN_member",
        "has_nuclear_weapons", "population_over_50M", "population_over_100M",
        "has_official_english",
    },
    "israel": {
        "is_in_asia", "has_coastline",
        "is_UN_member",
        "has_nuclear_weapons", "is_democracy",
    },
    "saudi_arabia": {
        "is_in_asia", "has_coastline",
        "is_UN_member",
        "gdp_top_20", "is_monarchy",
    },
    "iran": {
        "is_in_asia", "has_coastline",
        "is_UN_member",
        "population_over_50M",
    },
    "singapore": {
        "is_in_asia", "has_coastline", "is_island_nation",
        "is_UN_member",
        "is_democracy", "has_official_english",
    },
    "russia": {
        "is_in_europe", "is_in_asia", "has_coastline",
        "is_UN_member",
        "has_nuclear_weapons", "population_over_50M", "population_over_100M",
        "gdp_top_20",
    },
    "turkey": {
        "is_in_asia", "has_coastline",
        "is_UN_member", "is_NATO_member",
        "population_over_50M",
        "gdp_top_20",
    },

    # ── Americas ─────────────────────────────────────────────────────────
    "united_states": {
        "has_coastline",
        "is_UN_member", "is_NATO_member", "is_G7",
        "has_nuclear_weapons", "population_over_50M", "population_over_100M",
        "gdp_top_20", "is_democracy",
    },
    "canada": {
        "has_coastline",
        "is_UN_member", "is_NATO_member", "is_G7",
        "gdp_top_20", "is_democracy", "is_monarchy", "has_official_english",
    },
    "mexico": {
        "has_coastline",
        "is_UN_member",
        "population_over_50M", "population_over_100M",
        "gdp_top_20", "is_democracy",
    },
    "brazil": {
        "has_coastline",
        "is_UN_member",
        "population_over_50M", "population_over_100M",
        "gdp_top_20", "is_democracy",
    },
    "argentina": {
        "has_coastline",
        "is_UN_member",
        "is_democracy",
    },
    "colombia": {
        "has_coastline",
        "is_UN_member",
        "population_over_50M",
        "is_democracy",
    },
    "chile": {
        "has_coastline",
        "is_UN_member",
        "is_democracy",
    },
    "cuba": {
        "has_coastline", "is_island_nation",
        "is_UN_member",
    },
    "peru": {
        "has_coastline",
        "is_UN_member",
        "is_democracy",
    },
    "venezuela": {
        "has_coastline",
        "is_UN_member",
    },

    # ── Africa ───────────────────────────────────────────────────────────
    "south_africa": {
        "has_coastline",
        "is_UN_member",
        "population_over_50M",
        "is_democracy", "has_official_english",
    },
    "nigeria": {
        "has_coastline",
        "is_UN_member",
        "population_over_50M", "population_over_100M",
        "is_democracy", "has_official_english",
    },
    "egypt": {
        "has_coastline",
        "is_UN_member",
        "population_over_50M", "population_over_100M",
    },
    "kenya": {
        "has_coastline",
        "is_UN_member",
        "population_over_50M",
        "is_democracy", "has_official_english",
    },
    "ethiopia": {
        "is_UN_member",
        "population_over_50M", "population_over_100M",
    },
    "morocco": {
        "has_coastline",
        "is_UN_member",
        "is_monarchy",
    },
    "algeria": {
        "has_coastline",
        "is_UN_member",
    },

    # ── Oceania ──────────────────────────────────────────────────────────
    "australia": {
        "has_coastline",
        "is_UN_member",
        "gdp_top_20", "is_democracy", "is_monarchy", "has_official_english",
    },
    "new_zealand": {
        "has_coastline", "is_island_nation",
        "is_UN_member",
        "is_democracy", "is_monarchy", "has_official_english",
    },
}

COUNTRIES_NOTES = {
    "turkey/is_in_europe": (
        "Transcontinental; ~3% in Europe (Thrace), ~97% in Asia. "
        "Coded NO for is_in_europe—geographic majority in Asia."
    ),
    "turkey/is_democracy": (
        "Formally democratic with elections. EIU 2024 classifies as "
        "hybrid regime (score ~4.4). Coded NO."
    ),
    "russia/is_in_europe": (
        "Transcontinental; capital Moscow in Europe, ~23% of territory "
        "in Europe. Coded YES—conventionally European in geopolitical context."
    ),
    "russia/is_in_asia": (
        "Transcontinental; ~77% of territory in Asia. Coded YES."
    ),
    "united_states/has_official_english": (
        "No official language at federal level. English is de facto "
        "but not de jure. Coded NO."
    ),
    "israel/has_nuclear_weapons": (
        "Widely believed to possess ~90 warheads but never officially "
        "confirmed ('nuclear ambiguity' policy). Coded YES per "
        "intelligence consensus."
    ),
    "ireland/is_island_nation": (
        "Republic of Ireland shares island with Northern Ireland (UK). "
        "Conventionally treated as island nation. Coded YES."
    ),
    "australia/is_island_nation": (
        "Continent-island. Conventionally classified as continent, "
        "not island nation. Coded NO."
    ),
    "japan/is_monarchy": (
        "Constitutional monarchy; Emperor is ceremonial head of state. "
        "Coded YES."
    ),
    "canada/is_monarchy": (
        "Constitutional monarchy; King Charles III as head of state. "
        "Coded YES."
    ),
    "spain/population_over_50M": (
        "~48M in 2024, just under 50M threshold. Coded NO."
    ),
    "thailand/is_democracy": (
        "Civilian government restored after 2023 elections. "
        "Military influence persists. EIU 2024 ~6.5 (flawed democracy). "
        "Coded YES."
    ),
    "singapore/is_democracy": (
        "One-party dominance (PAP) since independence. EIU 2024: "
        "flawed democracy (~6.0). Coded YES."
    ),
    "pakistan/has_official_english": (
        "English is co-official per Article 251 of constitution "
        "(alongside Urdu). Coded YES."
    ),
    "nigeria/is_democracy": (
        "Regular elections with civilian government since 1999. "
        "Significant governance challenges. EIU 2024 ~4.1 (hybrid regime). "
        "Coded YES—borderline decision; multiparty elections are held."
    ),
    "switzerland/gdp_top_20": (
        "~$900B GDP, roughly #20 globally. Edge case at the boundary. "
        "Coded YES."
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# D1-30: COUNTRIES 30-attribute extension (additional 15 attrs)
# ═════════════════════════════════════════════════════════════════════════════
# Extra attributes: landlocked, is_republic, has_universal_healthcare,
#   is_OECD_member, has_space_program, is_commonwealth, has_death_penalty,
#   is_NATO_founder, has_high_speed_rail, is_BRICS_member,
#   has_tropical_climate, is_federal_state, uses_euro,
#   has_conscription, is_OPEC_member

COUNTRIES_30_EXTRA: dict[str, set[str]] = {
    # fmt: {country: set of EXTRA attrs that are TRUE}
    # landlocked, is_republic, has_universal_healthcare, is_OECD_member,
    # has_space_program, is_commonwealth, has_death_penalty, is_NATO_founder,
    # has_high_speed_rail, is_BRICS_member, has_tropical_climate,
    # is_federal_state, uses_euro, has_conscription, is_OPEC_member
    "algeria":        {"is_republic", "has_conscription", "is_OPEC_member"},
    "argentina":      {"is_republic", "is_federal_state", "has_conscription"},
    "australia":      {"is_commonwealth", "is_OECD_member", "has_universal_healthcare", "is_federal_state"},
    "austria":        {"landlocked", "is_republic", "has_universal_healthcare", "is_OECD_member", "uses_euro", "has_conscription"},
    "belgium":        {"has_universal_healthcare", "is_OECD_member", "is_NATO_founder", "has_high_speed_rail", "uses_euro", "is_federal_state"},
    "brazil":         {"is_republic", "has_space_program", "is_BRICS_member", "has_tropical_climate", "is_federal_state"},
    "canada":         {"is_commonwealth", "is_OECD_member", "has_universal_healthcare", "is_federal_state"},
    "chile":          {"is_republic", "is_OECD_member"},
    "china":          {"is_republic", "has_space_program", "has_death_penalty", "is_BRICS_member", "has_conscription"},
    "colombia":       {"is_republic", "has_tropical_climate"},
    "cuba":           {"is_republic", "has_tropical_climate"},
    "denmark":        {"has_universal_healthcare", "is_OECD_member", "is_NATO_founder"},
    "egypt":          {"is_republic", "has_death_penalty", "has_conscription"},
    "ethiopia":       {"landlocked", "is_republic", "is_BRICS_member", "has_tropical_climate", "is_federal_state"},
    "finland":        {"is_republic", "has_universal_healthcare", "is_OECD_member", "uses_euro", "has_conscription"},
    "france":         {"is_republic", "has_universal_healthcare", "is_OECD_member", "is_NATO_founder", "has_space_program", "has_high_speed_rail", "uses_euro"},
    "germany":        {"is_republic", "has_universal_healthcare", "is_OECD_member", "has_high_speed_rail", "uses_euro", "is_federal_state"},
    "greece":         {"is_republic", "has_universal_healthcare", "is_OECD_member", "uses_euro", "has_conscription"},
    "india":          {"is_republic", "has_space_program", "is_commonwealth", "has_death_penalty", "is_BRICS_member", "has_tropical_climate", "is_federal_state"},
    "indonesia":      {"is_republic", "has_tropical_climate"},
    "iran":           {"is_republic", "has_death_penalty", "is_OPEC_member", "has_conscription"},
    "ireland":        {"is_republic", "has_universal_healthcare", "is_OECD_member", "uses_euro"},
    "israel":         {"is_republic", "has_universal_healthcare", "is_OECD_member", "has_space_program", "has_conscription"},
    "italy":          {"is_republic", "has_universal_healthcare", "is_OECD_member", "is_NATO_founder", "has_high_speed_rail", "uses_euro"},
    "japan":          {"has_universal_healthcare", "is_OECD_member", "has_space_program", "has_high_speed_rail"},
    "kenya":          {"is_republic", "is_commonwealth", "has_tropical_climate"},
    "mexico":         {"is_republic", "is_OECD_member", "has_death_penalty", "is_federal_state"},
    "morocco":        {"has_conscription"},
    "netherlands":    {"has_universal_healthcare", "is_OECD_member", "is_NATO_founder", "has_high_speed_rail", "uses_euro"},
    "new_zealand":    {"is_commonwealth", "is_OECD_member", "has_universal_healthcare"},
    "nigeria":        {"is_republic", "is_commonwealth", "has_death_penalty", "has_tropical_climate", "is_federal_state", "is_OPEC_member"},
    "norway":         {"has_universal_healthcare", "is_OECD_member", "is_NATO_founder", "has_conscription"},
    "pakistan":        {"is_republic", "is_commonwealth", "has_death_penalty", "is_federal_state"},
    "peru":           {"is_republic"},
    "philippines":    {"is_republic", "has_death_penalty", "has_tropical_climate"},
    "poland":         {"is_republic", "has_universal_healthcare", "is_OECD_member"},
    "portugal":       {"is_republic", "has_universal_healthcare", "is_OECD_member", "is_NATO_founder", "uses_euro"},
    "russia":         {"is_republic", "has_space_program", "has_death_penalty", "is_BRICS_member", "is_federal_state", "has_conscription"},
    "saudi_arabia":   {"has_death_penalty", "is_OPEC_member"},
    "singapore":      {"is_republic", "is_commonwealth", "has_death_penalty"},
    "south_africa":   {"is_republic", "is_commonwealth", "is_BRICS_member"},
    "south_korea":    {"is_republic", "is_OECD_member", "has_high_speed_rail", "has_conscription"},
    "spain":          {"has_universal_healthcare", "is_OECD_member", "has_high_speed_rail", "uses_euro"},
    "sweden":         {"has_universal_healthcare", "is_OECD_member"},
    "switzerland":    {"landlocked", "is_republic", "has_universal_healthcare", "is_OECD_member", "is_federal_state"},
    "thailand":       {"has_tropical_climate", "has_conscription"},
    "turkey":         {"is_republic", "is_OECD_member", "has_conscription"},
    "united_kingdom": {"is_commonwealth", "is_OECD_member", "has_universal_healthcare", "is_NATO_founder", "has_high_speed_rail"},
    "united_states":  {"is_republic", "is_OECD_member", "has_space_program", "has_death_penalty", "is_NATO_founder", "is_federal_state"},
    "venezuela":      {"is_republic", "has_tropical_climate", "is_OPEC_member"},
}

COUNTRIES_30_NOTES = {
    **COUNTRIES_NOTES,
    "austria/landlocked": "Landlocked, no coastline. Coded YES.",
    "switzerland/landlocked": "Landlocked, no coastline. Coded YES.",
    "ethiopia/landlocked": "Landlocked since Eritrea independence (1993). Not coded as landlocked because 'landlocked' was not in the original 15 attrs and Ethiopia lacks 'has_coastline'. In 30-attr version, Ethiopia IS landlocked (no has_coastline, YES landlocked).",
    "japan/is_republic": "Constitutional monarchy with Emperor. Coded NO (not a republic).",
    "united_kingdom/is_republic": "Constitutional monarchy. Coded NO.",
    "united_states/has_death_penalty": "Federal death penalty exists; most executions at state level. Coded YES.",
    "russia/has_death_penalty": "Moratorium since 1996 but not abolished. Coded YES (on books).",
    "india/is_BRICS_member": "Founding BRICS member (2006). Coded YES.",
    "saudi_arabia/is_OPEC_member": "Founding OPEC member (1960). Coded YES.",
    "turkey/is_NATO_founder": "Joined NATO 1952, not founding 1949. Coded NO... wait, Turkey joined in 1952. Coded NO for is_NATO_founder.",
}

def build_countries_30_gold() -> dict[str, set[str]]:
    """Merge 15-attr gold + 15 extra attrs → 30-attr gold."""
    merged = {}
    for name, base_attrs in COUNTRIES_GOLD.items():
        extra = COUNTRIES_30_EXTRA.get(name, set())
        merged[name] = base_attrs | extra
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# D2: ANIMALS — 40 animals × 15 attributes
# ═════════════════════════════════════════════════════════════════════════════
# Attributes: is_mammal, is_bird, is_reptile, is_fish,
#   can_fly, lives_in_water, is_domestic, is_carnivore,
#   is_herbivore, has_fur, lays_eggs, is_nocturnal,
#   is_endangered, lives_in_groups, larger_than_human

ANIMALS_GOLD: dict[str, set[str]] = {
    # ── Mammals ──────────────────────────────────────────────────────────
    "dog": {
        "is_mammal", "is_domestic", "is_carnivore",
        "has_fur", "lives_in_groups",
    },
    "cat": {
        "is_mammal", "is_domestic", "is_carnivore",
        "has_fur", "is_nocturnal",
    },
    "horse": {
        "is_mammal", "is_domestic", "is_herbivore",
        "has_fur", "lives_in_groups", "larger_than_human",
    },
    "cow": {
        "is_mammal", "is_domestic", "is_herbivore",
        "has_fur", "lives_in_groups", "larger_than_human",
    },
    "lion": {
        "is_mammal", "is_carnivore",
        "has_fur", "is_nocturnal", "lives_in_groups", "larger_than_human",
    },
    "elephant": {
        "is_mammal", "is_herbivore",
        "is_endangered", "lives_in_groups", "larger_than_human",
    },
    "whale": {
        "is_mammal", "lives_in_water", "is_carnivore",
        "is_endangered", "lives_in_groups", "larger_than_human",
    },
    "bat": {
        "is_mammal", "can_fly",
        "has_fur", "is_nocturnal", "lives_in_groups",
    },
    "tiger": {
        "is_mammal", "is_carnivore",
        "has_fur", "is_nocturnal", "is_endangered",
    },
    "bear": {
        "is_mammal", "is_carnivore",
        "has_fur", "larger_than_human",
    },
    "wolf": {
        "is_mammal", "is_carnivore",
        "has_fur", "is_nocturnal", "lives_in_groups",
    },
    "rabbit": {
        "is_mammal", "is_domestic", "is_herbivore",
        "has_fur", "lives_in_groups",
    },
    "dolphin": {
        "is_mammal", "lives_in_water", "is_carnivore",
        "lives_in_groups", "larger_than_human",
    },
    "platypus": {
        "is_mammal", "lives_in_water", "is_carnivore",
        "has_fur", "lays_eggs", "is_nocturnal",
    },
    "mouse": {
        "is_mammal", "is_domestic",
        "has_fur", "is_nocturnal", "lives_in_groups",
    },
    "koala": {
        "is_mammal", "is_herbivore",
        "has_fur", "is_nocturnal", "is_endangered",
    },
    "gorilla": {
        "is_mammal", "is_herbivore",
        "has_fur", "is_endangered", "lives_in_groups", "larger_than_human",
    },
    "deer": {
        "is_mammal", "is_herbivore",
        "has_fur", "lives_in_groups",
    },
    "cheetah": {
        "is_mammal", "is_carnivore",
        "has_fur", "is_endangered",
    },
    "pig": {
        "is_mammal", "is_domestic",
        "lives_in_groups", "larger_than_human",
    },
    "giraffe": {
        "is_mammal", "is_herbivore",
        "has_fur", "is_endangered", "lives_in_groups", "larger_than_human",
    },

    # ── Birds ────────────────────────────────────────────────────────────
    "eagle": {
        "is_bird", "can_fly", "is_carnivore",
        "lays_eggs",
    },
    "penguin": {
        "is_bird", "lives_in_water", "is_carnivore",
        "lays_eggs", "lives_in_groups",
    },
    "chicken": {
        "is_bird", "is_domestic",
        "lays_eggs", "lives_in_groups",
    },
    "owl": {
        "is_bird", "can_fly", "is_carnivore",
        "lays_eggs", "is_nocturnal",
    },
    "parrot": {
        "is_bird", "can_fly", "is_domestic", "is_herbivore",
        "lays_eggs", "lives_in_groups",
    },
    "ostrich": {
        "is_bird", "is_herbivore",
        "lays_eggs", "lives_in_groups", "larger_than_human",
    },
    "swan": {
        "is_bird", "can_fly", "lives_in_water", "is_herbivore",
        "lays_eggs",
    },
    "pigeon": {
        "is_bird", "can_fly", "is_domestic", "is_herbivore",
        "lays_eggs", "lives_in_groups",
    },

    # ── Reptiles ─────────────────────────────────────────────────────────
    "crocodile": {
        "is_reptile", "lives_in_water", "is_carnivore",
        "lays_eggs", "is_nocturnal", "larger_than_human",
    },
    "snake": {
        "is_reptile", "is_carnivore",
        "lays_eggs", "is_nocturnal",
    },
    "turtle": {
        "is_reptile", "lives_in_water",
        "lays_eggs", "is_endangered",
    },
    "lizard": {
        "is_reptile", "is_carnivore",
        "lays_eggs",
    },
    "komodo_dragon": {
        "is_reptile", "is_carnivore",
        "lays_eggs", "is_endangered",
    },
    "chameleon": {
        "is_reptile", "is_carnivore",
        "lays_eggs",
    },

    # ── Fish ─────────────────────────────────────────────────────────────
    "shark": {
        "is_fish", "lives_in_water", "is_carnivore",
        "is_endangered", "larger_than_human",
    },
    "salmon": {
        "is_fish", "lives_in_water", "is_carnivore",
        "lays_eggs", "is_endangered", "lives_in_groups",
    },
    "goldfish": {
        "is_fish", "lives_in_water", "is_domestic",
        "lays_eggs", "lives_in_groups",
    },
    "tuna": {
        "is_fish", "lives_in_water", "is_carnivore",
        "lays_eggs", "is_endangered", "lives_in_groups", "larger_than_human",
    },
    "seahorse": {
        "is_fish", "lives_in_water",
        "is_endangered",
    },
}

ANIMALS_NOTES = {
    "whale/is_fish": (
        "Whale is a mammal, not a fish. Common misconception. "
        "Coded is_fish=NO, is_mammal=YES."
    ),
    "bat/can_fly": (
        "Only mammal capable of true sustained flight. "
        "Flying squirrels glide but don't fly. Coded YES."
    ),
    "platypus/lays_eggs": (
        "Monotreme—mammal that lays eggs. One of only 5 extant "
        "monotreme species. Coded is_mammal=YES AND lays_eggs=YES."
    ),
    "platypus/is_carnivore": (
        "Eats insects, larvae, shrimp—invertebrate predator. "
        "Coded YES (animal food sources)."
    ),
    "penguin/can_fly": (
        "Flightless bird; wings adapted as flippers for swimming. "
        "Coded can_fly=NO."
    ),
    "elephant/has_fur": (
        "Very sparse bristles/hair, not conventionally 'fur'. "
        "Coded NO."
    ),
    "whale/has_fur": (
        "Virtually hairless aquatic mammal. Some species have sparse "
        "hairs at birth. Coded NO."
    ),
    "dolphin/has_fur": (
        "Hairless aquatic mammal. Coded NO."
    ),
    "pig/has_fur": (
        "Sparse bristles/coarse hair, not conventionally 'fur'. "
        "Coded NO."
    ),
    "pig/is_herbivore": (
        "Omnivore—eats both plant and animal matter. "
        "Coded is_carnivore=NO, is_herbivore=NO."
    ),
    "bear/is_carnivore": (
        "Order Carnivora but many species omnivorous (brown bear, "
        "black bear). Polar bear is hypercarnivore. "
        "Coded YES (taxonomic classification + meat-eating capacity)."
    ),
    "cat/is_nocturnal": (
        "Technically crepuscular (active dawn/dusk), commonly "
        "considered nocturnal. Coded YES."
    ),
    "dog/is_carnivore": (
        "Order Carnivora, primarily meat-eating but can digest "
        "plant matter. Coded YES."
    ),
    "snake/lays_eggs": (
        "Most snakes are oviparous. Some (boas, vipers) are "
        "viviparous. Coded YES (majority case)."
    ),
    "shark/lays_eggs": (
        "Most sharks are viviparous (live birth). Some species "
        "(horn shark, cat shark) lay egg cases. Coded NO (majority case)."
    ),
    "seahorse/lays_eggs": (
        "Female deposits eggs into male's brood pouch where they "
        "develop. Male 'gives birth'. Unique reproduction. Coded NO."
    ),
    "mouse/is_domestic": (
        "Wild mice are common, but domestic/pet mice are widespread. "
        "Coded YES due to extensive domestication history."
    ),
    "australia_note/is_island_nation": (
        "In the countries domain: Australia is a continent-island, "
        "conventionally classified as continent. Coded NO."
    ),
    "ostrich/is_herbivore": (
        "Primarily herbivorous (plants, seeds) but occasionally eats "
        "insects/small animals. Coded YES (primary diet)."
    ),
    "turtle/is_endangered": (
        "Many sea turtle species endangered (green, hawksbill, "
        "leatherback). Generic 'turtle' represents the taxon. Coded YES."
    ),
    "dolphin/larger_than_human": (
        "Bottlenose dolphin: 200-600kg, 2-4m. Clearly larger than "
        "average human by mass. Coded YES."
    ),
    "pig/larger_than_human": (
        "Domestic farm pig: 100-350kg. Market weight ~100-130kg. "
        "Coded YES."
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# Build & Save
# ═════════════════════════════════════════════════════════════════════════════

def compute_canonical_basis(
    attributes: list[str],
    gold: dict[str, set[str]],
) -> list[dict]:
    """Compute canonical basis using FCA exploration."""
    oracle = GoldOracle(gold)
    result = full_exploration(attributes, oracle, max_iterations=100_000)
    return [
        {
            "premise": sorted(impl.premise),
            "conclusion": sorted(impl.conclusion),
        }
        for impl in result.implications
    ]


def build_json(
    domain: dict,
    gold: dict[str, set[str]],
    notes: dict[str, str],
    source: str,
) -> dict:
    """Build complete gold standard JSON."""
    attrs = domain["attributes"]

    # Validate: all object attributes must be in the domain's attribute list
    attr_set = set(attrs)
    for name, obj_attrs in gold.items():
        invalid = obj_attrs - attr_set
        assert not invalid, f"{name} has unknown attrs: {invalid}"

    basis = compute_canonical_basis(attrs, gold)
    print(f"  {domain['name']}: {len(gold)} objects, {len(basis)} implications")
    for b in basis:
        p = ", ".join(b["premise"]) or "∅"
        c = ", ".join(sorted(set(b["conclusion"]) - set(b["premise"])))
        print(f"    {{{p}}} → {{{c}}}")

    return {
        "domain": domain["name"],
        "description": domain["description"],
        "attributes": attrs,
        "source": source,
        "date": "2026-03-22",
        "num_objects": len(gold),
        "num_attributes": len(attrs),
        "objects": {
            name: sorted(attrs)
            for name, attrs in sorted(gold.items())
        },
        "ambiguity_notes": dict(sorted(notes.items())),
        "canonical_basis": basis,
        "num_implications": len(basis),
    }


def main():
    out_dir = Path(__file__).resolve().parent

    print("Building gold standards...")

    # Countries
    countries = build_json(
        COUNTRIES_DOMAIN, COUNTRIES_GOLD, COUNTRIES_NOTES,
        source="Curated from Wikidata, CIA World Factbook, EIU Democracy Index (2024)",
    )
    with open(out_dir / "countries.json", "w") as f:
        json.dump(countries, f, indent=2, ensure_ascii=False)
    print(f"  → {out_dir / 'countries.json'}")

    # Countries 30-attr
    countries_30_gold = build_countries_30_gold()
    countries_30 = build_json(
        COUNTRIES_30, countries_30_gold, COUNTRIES_30_NOTES,
        source="Curated from Wikidata, CIA World Factbook, EIU, Wikipedia (2024)",
    )
    with open(out_dir / "countries_30.json", "w") as f:
        json.dump(countries_30, f, indent=2, ensure_ascii=False)
    print(f"  → {out_dir / 'countries_30.json'}")

    # Animals
    animals = build_json(
        ANIMALS_DOMAIN, ANIMALS_GOLD, ANIMALS_NOTES,
        source="Curated from IUCN Red List, Encyclopedia of Life, expert review (2024)",
    )
    with open(out_dir / "animals.json", "w") as f:
        json.dump(animals, f, indent=2, ensure_ascii=False)
    print(f"  → {out_dir / 'animals.json'}")

    print("Done.")


if __name__ == "__main__":
    main()
