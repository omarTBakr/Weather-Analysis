"""Continents and country-name mappings."""

from __future__ import annotations


class Geography:
    """Continent ordering / codes and the fixes needed to map messy country names onto continents."""

    # canonical order used for plotting AND for the integer `continent_code` feature
    CONTINENT_ORDER = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
    # unknown continents are encoded as -1
    CONTINENT_CODES = {name: code for code, name in enumerate(CONTINENT_ORDER)}
    # latitude bands (degrees) of the climatology figures, south to north, and their labels
    LATITUDE_BAND_EDGES = [-90, -35, -23.5, 0, 23.5, 35, 50, 90]
    LATITUDE_BAND_LABELS = ["<35°S", "35–23.5°S", "23.5°S–0", "0–23.5°N", "23.5–35°N", "35–50°N", ">50°N"]

    # pycountry-convert continent code -> continent name
    ISO_CONTINENT_NAMES = {
        "AF": "Africa",
        "AS": "Asia",
        "EU": "Europe",
        "NA": "North America",
        "SA": "South America",
        "OC": "Oceania",
        "AN": "Antarctica",
    }

    # countries pycountry-convert cannot resolve (or resolves to a non-continent code)
    CONTINENT_OVERRIDES = {
        "Timor-Leste": "Asia",
        "East Timor": "Asia",
        "Vatican City": "Europe",
        "Kosovo": "Europe",
        "Bolivia": "South America",
        "Venezuela": "South America",
        "Iran": "Asia",
        "Syria": "Asia",
        "Laos": "Asia",
        "Vietnam": "Asia",
        "Russia": "Europe",
        "South Korea": "Asia",
        "North Korea": "Asia",
        "Tanzania": "Africa",
        "Moldova": "Europe",
        "Macedonia": "Europe",
        "North Macedonia": "Europe",
        "Democratic Republic of Congo": "Africa",
        "Congo": "Africa",
        "Ivory Coast": "Africa",
        "Cote d'Ivoire": "Africa",
        "Swaziland": "Africa",
        "Eswatini": "Africa",
        "Cape Verde": "Africa",
        "Micronesia": "Oceania",
        "Palestine": "Asia",
        "Turkey": "Asia",
        "Türkiye": "Asia",
        "Bahamas": "North America",
        "Gambia": "Africa",
        "Saint Vincent and the Grenadines": "North America",
        "Saint Kitts and Nevis": "North America",
        "Saint Lucia": "North America",
        "Brunei Darussalam": "Asia",
        "Brunei": "Asia",
        "Myanmar": "Asia",
        "Burma": "Asia",
        "Guinea-Bissau": "Africa",
        "Marshall Islands": "Oceania",
        "Solomon Islands": "Oceania",
        "Western Sahara": "Africa",
        "Fiji Islands": "Oceania",
        "Kyrghyzstan": "Asia",
        "Seychelles Islands": "Africa",
        "Antarctica": "Antarctica",
    }

    # non-English / variant spellings that appear as a location's dominant country
    COUNTRY_FIXES = {
        "USA United States of America": "United States of America",
        "United States": "United States of America",
        "Turkey": "Türkiye",
    }
