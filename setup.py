from setuptools import setup

setup(
    name='mu',
    version='0.0.02',
    license='GPL',
    description='Python-representation of classical musical content.',
    author='Levin Eric Zimmermann',
    author_email='levin-eric.zimmermann@folkwang-uni.de',
    url='https://github.com/uummoo/mu',
    packages=['mu', 'mu.abstract', 'mu.time', 'mu.rhy', 'mu.mel', 'mu.sco'],
    setup_requires=[''],
    tests_require=['nosetests'],
    install_requires=['pyprimes>= 0.2.2a0'],
    extras_require={'Lilypond':  ["Music21>=4.1.0"]},
    python_requires='>=3.6'
)
