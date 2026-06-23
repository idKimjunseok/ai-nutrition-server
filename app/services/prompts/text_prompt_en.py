from __future__ import annotations

"""텍스트 음식명 분석용 영어 프롬프트."""


def build(food_text: str) -> str:
    return (
        "For each item in the following food list, estimate the calories and macronutrients "
        "for one typical serving and respond ONLY in the following JSON array format. "
        "Do not include any explanatory text. "
        "If the items are separated by commas or spaces, treat each as a separate entry. "
        "If nothing in the input can be recognized as food, respond with an empty array []. "
        "Write all food names and portion descriptions in English. "
        "All numeric values must be plain numbers without units.\n\n"
        f"Food: {food_text}\n\n"
        "[\n"
        "  {\n"
        '    "food_name": "Food name in English",\n'
        '    "calories_kcal": calorie_number,\n'
        '    "portion_description": "Estimated serving size",\n'
        '    "carbs_g": carbohydrate_grams_number,\n'
        '    "protein_g": protein_grams_number,\n'
        '    "fat_g": fat_grams_number,\n'
        '    "confidence": confidence_number_between_0_and_1\n'
        "  }\n"
        "]"
    )
