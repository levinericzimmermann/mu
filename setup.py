from setuptools import setup

setup(
    name='mu',
    version='0.0.02',
    license='GPL',
    description='Python-representation of classical musical content.',
    author='Levin Eric Zimmermann',
    author_email='levin-eric.zimmermann@folkwang-uni.de',
    url='https://github.com/uummoo/mu',
    packages=['mu', 'mu.abstract', 'mu.time', 'mu.rhy',
              'mu.mel', 'mu.sco', 'mu.utils'],
    package_data={'': ['mu/utils/primes.json'], '': ['mu/mel/12edo']},
    include_package_data=True,
    setup_requires=[''],
    tests_require=['nose'],
    install_requires=[''],
    extras_require={},
    python_requires='>=3.6'
)
