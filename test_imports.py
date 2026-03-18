def check(module_name: str) -> None:
    try:
        __import__(module_name)
        print(f"{module_name}: OK")
    except Exception as exc:
        print(f"{module_name}: ERROR -> {exc}")


if __name__ == "__main__":
    for module in [
        "streamlit",
        "pandas",
        "fitz",
        "requests",
        "tenacity",
        "dotenv",
    ]:
        check(module)
