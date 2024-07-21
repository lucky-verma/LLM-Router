from setuptools import setup, find_packages

setup(
    name='multi_model_llm_system',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'transformers',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'data_preparation=scripts.data_preparation:main',
            'fine_tune_model=scripts.fine_tune_model:main',
            'train_router=scripts.train_router:main',
            'inference=scripts.inference:main',
            'evaluate=scripts.evaluate:main',
        ],
    },
)
