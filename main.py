from dotenv import load_dotenv
import os

load_dotenv()

def main() -> None:
    print(os.environ.get("OPENROUTER_API_KEY"))


if __name__ == "__main__":
    main()
