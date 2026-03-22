"""
Domain Definitions — Hallucination Benchmarks
==============================================
Each domain serves as a benchmark for measuring SLM consistency,
NOT as an application demo. Domains are ordered by difficulty:
D1 (factual) < D2 (commonsense) < D3 (specialized).
"""

# --- D0: Toy Fruit (internal validation only, never in paper) ---
FRUIT_DOMAIN = {
    "name": "fruits",
    "description": "properties of common fruits",
    "attributes": ["red", "sweet", "has_seed"],
    "initial_examples": {},
    "gold_objects": {
        "apple":      {"red", "sweet", "has_seed"},
        "tomato":     {"red", "has_seed"},
        "banana":     {"sweet"},
        "watermelon": {"sweet", "has_seed"},
        "lemon":      {"has_seed"},
        "strawberry": {"red", "sweet", "has_seed"},
    },
}

# --- D1: Countries (main factual benchmark) ---
COUNTRIES_DOMAIN = {
    "name": "countries",
    "description": "geopolitical properties of well-known countries",
    "attributes": [
        "is_in_europe", "is_in_asia", "has_coastline", "is_island_nation",
        "is_UN_member", "is_NATO_member", "is_EU_member", "is_G7",
        "has_nuclear_weapons", "population_over_50M", "population_over_100M",
        "gdp_top_20", "is_democracy", "is_monarchy", "has_official_english",
    ],
    "initial_examples": {},
    "gold_standard_path": "gold_standards/countries.json",
}

# --- D2: Animals (commonsense benchmark) ---
ANIMALS_DOMAIN = {
    "name": "animals",
    "description": "biological and behavioral properties of common animals",
    "attributes": [
        "is_mammal", "is_bird", "is_reptile", "is_fish",
        "can_fly", "lives_in_water", "is_domestic", "is_carnivore",
        "is_herbivore", "has_fur", "lays_eggs", "is_nocturnal",
        "is_endangered", "lives_in_groups", "larger_than_human",
    ],
    "initial_examples": {},
    "gold_standard_path": "gold_standards/animals.json",
    "notes": "Edge cases: platypus(mammal+eggs), penguin(bird+no fly), bat(mammal+fly)",
}

# --- D3: SE Concepts (domain-specific benchmark) ---
SE_DOMAIN = {
    "name": "se_concepts",
    "description": "systems engineering entity types based on BFO/ISO 21838-2",
    "attributes": [
        "is_continuant", "is_occurrent", "is_independent", "is_dependent",
        "has_function", "has_requirement", "is_physical", "is_informational",
        "is_verifiable", "is_decomposable", "has_interface", "is_reusable",
    ],
    "initial_examples": {},
    "gold_standard_path": "gold_standards/se_concepts.json",
}

# --- Domain Registry ---
DOMAINS = {
    "fruits": FRUIT_DOMAIN,
    "countries": COUNTRIES_DOMAIN,
    "animals": ANIMALS_DOMAIN,
    "se_concepts": SE_DOMAIN,
}

# --- Scaling variants of D1 for Exp 5 ---
COUNTRIES_10 = {**COUNTRIES_DOMAIN, "name": "countries_10",
    "attributes": COUNTRIES_DOMAIN["attributes"][:10]}
COUNTRIES_15 = {**COUNTRIES_DOMAIN, "name": "countries_15",
    "attributes": COUNTRIES_DOMAIN["attributes"][:15]}
COUNTRIES_20 = {**COUNTRIES_DOMAIN, "name": "countries_20",
    "attributes": COUNTRIES_DOMAIN["attributes"][:15] + [
        "landlocked", "is_republic", "has_universal_healthcare",
        "is_OECD_member", "has_space_program"]}
COUNTRIES_30 = {**COUNTRIES_DOMAIN, "name": "countries_30",
    "attributes": COUNTRIES_DOMAIN["attributes"][:15] + [
        "landlocked", "is_republic", "has_universal_healthcare",
        "is_OECD_member", "has_space_program",
        "is_commonwealth", "has_death_penalty", "is_NATO_founder",
        "has_high_speed_rail", "is_BRICS_member",
        "has_tropical_climate", "is_federal_state", "uses_euro",
        "has_conscription", "is_OPEC_member"]}

SCALING_DOMAINS = {
    "countries_10": COUNTRIES_10,
    "countries_15": COUNTRIES_15,
    "countries_20": COUNTRIES_20,
    "countries_30": COUNTRIES_30,
}
