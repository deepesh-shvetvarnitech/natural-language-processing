
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "product_reviews.csv"

SAMPLES = [
    ("battery lasts all day and charges quickly", "electronics"),
    ("the phone battery life is excellent", "electronics"),
    ("camera takes sharp photos in low light", "electronics"),
    ("the screen is bright and easy to read", "electronics"),
    ("wireless headphones have clear sound and deep bass", "electronics"),
    ("the laptop runs fast and the keyboard feels comfortable", "electronics"),
    ("charging cable stopped working after two days", "electronics"),
    ("the speaker sound is loud but distorted", "electronics"),
    ("this moisturizer makes my skin soft and hydrated", "beauty"),
    ("face cream has a pleasant smell and absorbs quickly", "beauty"),
    ("the shampoo leaves my hair smooth and shiny", "beauty"),
    ("this sunscreen feels light and is not greasy", "beauty"),
    ("the lipstick color is beautiful and lasts long", "beauty"),
    ("the lotion caused irritation and dryness", "beauty"),
    ("comfortable running shoes with excellent grip", "clothing"),
    ("the shirt fabric is soft and fits perfectly", "clothing"),
    ("these jeans are durable and look stylish", "clothing"),
    ("the jacket is warm but too heavy", "clothing"),
    ("shoes are tight and uncomfortable for walking", "clothing"),
    ("the cotton dress is lightweight and breathable", "clothing"),
    ("coffee maker brews rich coffee in minutes", "home"),
    ("the vacuum cleaner picks up dust very well", "home"),
    ("this kitchen blender is powerful and easy to clean", "home"),
    ("the blanket is soft warm and cozy", "home"),
    ("the chair is sturdy and comfortable to sit on", "home"),
    ("the toaster burns bread and is difficult to use", "home"),
    ("fast delivery and the package arrived safely", "service"),
    ("customer support solved my issue quickly", "service"),
    ("the product arrived late and the box was damaged", "service"),
    ("refund process was simple and the seller was helpful", "service"),
]

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(SAMPLES, columns=["review", "category"])
    df.insert(0, "review_id", [f"R{i:03d}" for i in range(1, len(df) + 1)])
    df.to_csv(DATA_PATH, index=False)
    print(f"Created {DATA_PATH} with {len(df)} reviews.")

if __name__ == "__main__":
    main()
