from setuptools import setup


setup(
    name="data_driven_disturbance_estimator",
    version="0.0.1",
    author="Matthias Schöffel",
    author_email="matthiasschoeffel97@gmail.com",
    description=(
        "Uses data driven system representation and kernel density estimator to estimate disturbance on LTI-system."),
    license="BSD",
    keywords="data driven disturbance estimator",
    url="https://github.com/ratherbeflyin2080/data_driven_disturbance_estimator",
    packages=['data_driven_mpc', 'tests','disturbance_estimation','lti_system'],
    entry_points={
        'console_scripts': [
            'dsa = data_driven_mpc.main:main',
        ],
    },
)
