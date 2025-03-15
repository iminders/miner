from setuptools import setup, find_packages

setup(
    name="miner",
    version="0.1.0",
    description="A股高频因子挖掘系统",
    author="iminders",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        "numba",
        "h5py",
        "pyarrow",
        "dask",
        "lightgbm",
        "xgboost",
        "backtrader",
        "dash",
        "streamlit",
    ],
)