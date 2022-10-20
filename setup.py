from setuptools import setup


setup(
    name="data_driven_stochastic_mpc",
    version="0.0.1",
    author="Matthias Schöffel",
    author_email="matthiasschoeffel97@gmail.com",
    description=(
        "Uses data driven system representation, kernel density estimator and online constraint tightening to adapt to changing environment."),
    license="BSD",
    keywords="data driven stochastic model predictive control with online disturbance estimation and constraint tightening",
    url="https://github.com/m-schoeffel/data_driven_stochastic_mpc",
    packages=['data_driven_mpc','disturbance_estimation','lti_system','simulation','config','graphics','constraint_tightening','recorded_data'],
    entry_points={
        'console_scripts': [
            'run_ddsmpc = simulation.main:main',
        ],
    },
)
