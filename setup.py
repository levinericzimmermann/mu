from setuptools import setup

setup(
    name='mu',
    version='0.0.01',
    license='GPL',
    description='Python-representation of classical musical content.',
    author='Levin Eric Zimmermann',
    author_email='levin-eric.zimmermann@folkwang-uni.de',
    url='https://github.com/uummoo/',
    packages=['mu', 'mu.mel'],
    setup_requires=[''],
    tests_require=['nosetests'],
    python_requires='>=3.6'
)
