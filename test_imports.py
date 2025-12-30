try:
    import pandas
    print("Pandas imported successfully")
    print(f"Pandas version: {pandas.__version__}")
except ImportError as e:
    print(f"Error importing pandas: {e}")

try:
    import plotly.express as px
    print("Plotly Express imported successfully")
except ImportError as e:
    print(f"Error importing plotly.express: {e}")
