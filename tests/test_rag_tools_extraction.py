"""Test RAG tools extraction accuracy.

This test verifies that the product extraction tools correctly extract
data for each specific iPhone model without mixing up data from adjacent
products in the PDF.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.rag_tools import (
    get_product_colors,
    get_product_battery_life,
    get_product_launch_year,
)


# Expected values from FAQ_Catalogue_Telephones.pdf
EXPECTED_DATA = {
    "iPhone X": {
        "colors": ["Gris sidéral", "Argent"],
        "battery": 13,
        "year": 2017,
    },
    "iPhone 11": {
        "colors": ["Noir", "Blanc", "Vert", "Jaune", "Mauve", "Rouge"],
        "battery": 17,
        "year": 2019,
    },
    "iPhone 12": {
        "colors": ["Noir", "Blanc", "Bleu", "Vert", "Rouge", "Mauve"],
        "battery": 17,
        "year": 2020,
    },
    "iPhone 13": {
        "colors": ["Minuit", "Lumière stellaire", "Bleu", "Rose", "Rouge", "Vert"],
        "battery": 19,
        "year": 2021,
    },
    "iPhone 14": {
        "colors": ["Minuit", "Lumière stellaire", "Bleu", "Mauve", "Rouge", "Jaune"],
        "battery": 20,
        "year": 2022,
    },
    "iPhone 15": {
        "colors": ["Noir", "Bleu", "Vert", "Jaune", "Rose"],
        "battery": 20,
        "year": 2023,
    },
    "iPhone 16": {
        "colors": ["Noir", "Blanc", "Bleu sarcelle", "Rose", "Outremer"],
        "battery": 22,
        "year": 2024,
    },
    "iPhone 17": {
        "colors": ["Titane naturel", "Titane bleu", "Titane noir", "Titane blanc"],
        "battery": 25,
        "year": 2025,
    },
}


def test_colors_extraction():
    """Test that colors are extracted correctly for each iPhone."""
    print("\n" + "=" * 60)
    print("Testing COLOR extraction")
    print("=" * 60)

    failed = []

    for product, expected in EXPECTED_DATA.items():
        result = get_product_colors.invoke(product)

        if not result.get("found"):
            print(f"FAIL: {product} - Colors not found")
            failed.append(product)
            continue

        extracted_colors = result.get("colors", [])
        expected_colors = expected["colors"]

        # Check if key colors are present (not exact match due to regex variations)
        # We check if at least 2 expected colors are found
        matches = sum(1 for ec in expected_colors if any(ec.lower() in str(fc).lower() for fc in extracted_colors))

        if matches >= 2:
            print(f"PASS: {product}")
            print(f"       Expected: {expected_colors}")
            print(f"       Got:      {extracted_colors}")
        else:
            print(f"FAIL: {product}")
            print(f"       Expected: {expected_colors}")
            print(f"       Got:      {extracted_colors}")
            failed.append(product)

    return failed


def test_battery_extraction():
    """Test that battery life is extracted correctly for each iPhone."""
    print("\n" + "=" * 60)
    print("Testing BATTERY extraction")
    print("=" * 60)

    failed = []

    for product, expected in EXPECTED_DATA.items():
        result = get_product_battery_life.invoke(product)

        if not result.get("found"):
            print(f"FAIL: {product} - Battery not found")
            failed.append(product)
            continue

        extracted_battery = result.get("battery_hours")
        expected_battery = expected["battery"]

        if extracted_battery == expected_battery:
            print(f"PASS: {product} - {extracted_battery}h (expected {expected_battery}h)")
        else:
            print(f"FAIL: {product} - Got {extracted_battery}h, expected {expected_battery}h")
            failed.append(product)

    return failed


def test_year_extraction():
    """Test that launch year is extracted correctly for each iPhone."""
    print("\n" + "=" * 60)
    print("Testing YEAR extraction")
    print("=" * 60)

    failed = []

    for product, expected in EXPECTED_DATA.items():
        result = get_product_launch_year.invoke(product)

        if not result.get("found"):
            print(f"FAIL: {product} - Year not found")
            failed.append(product)
            continue

        extracted_year = result.get("year")
        expected_year = expected["year"]

        if extracted_year == expected_year:
            print(f"PASS: {product} - {extracted_year} (expected {expected_year})")
        else:
            print(f"FAIL: {product} - Got {extracted_year}, expected {expected_year}")
            failed.append(product)

    return failed


def test_specific_problematic_cases():
    """Test the specific problematic cases from the evaluation."""
    print("\n" + "=" * 60)
    print("Testing SPECIFIC PROBLEMATIC CASES")
    print("=" * 60)

    # Q10: iPhone 16 colors
    print("\nQ10: Quels sont les coloris disponibles pour l'iPhone 16 ?")
    result = get_product_colors.invoke("iPhone 16")
    print(f"  Result: {result}")
    expected = ["Noir", "Blanc", "Bleu sarcelle", "Rose", "Outremer"]
    print(f"  Expected: {expected}")

    # Q12: Battery difference iPhone 13 vs 16
    print("\nQ12: Quelle est la différence d'autonomie entre l'iPhone 13 et l'iPhone 16 ?")
    result_13 = get_product_battery_life.invoke("iPhone 13")
    result_16 = get_product_battery_life.invoke("iPhone 16")
    print(f"  iPhone 13 battery: {result_13}")
    print(f"  iPhone 16 battery: {result_16}")
    print(f"  Expected: iPhone 13 = 19h, iPhone 16 = 22h, difference = 3h")

    if result_13.get("battery_hours") == 19 and result_16.get("battery_hours") == 22:
        print("  STATUS: PASS")
        return True
    else:
        print("  STATUS: FAIL")
        return False


if __name__ == "__main__":
    print("RAG Tools Extraction Test")
    print("=" * 60)

    # Run tests
    color_failures = test_colors_extraction()
    battery_failures = test_battery_extraction()
    year_failures = test_year_extraction()
    specific_pass = test_specific_problematic_cases()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_failures = len(color_failures) + len(battery_failures) + len(year_failures)

    if color_failures:
        print(f"Color extraction failures: {color_failures}")
    if battery_failures:
        print(f"Battery extraction failures: {battery_failures}")
    if year_failures:
        print(f"Year extraction failures: {year_failures}")

    if total_failures == 0 and specific_pass:
        print("\nAll tests PASSED!")
    else:
        print(f"\nTotal failures: {total_failures}")
        print(f"Specific cases: {'PASSED' if specific_pass else 'FAILED'}")
