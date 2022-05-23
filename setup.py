from setuptools import setup, find_packages

VERSION = '0.0.3' 
DESCRIPTION = 'Discrete-event Quantum network simulator using error-basis model'
LONG_DESCRIPTION = 'Inspire from QuISP by AQUA, Keio university, qwanta is simplified version with simplicity of python at your hand.'

# Setting up
setup(
        name="qwanta", 
        version=VERSION,
        author="Poramet Pathumsoot",
        author_email="Poramet.path@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'networkx',
            'simpy',
            'pandas',
            'seaborn',
            'dill',
            'tqdm',
            'matplotlib',
            'pyvis',
            'numpy',
            'scipy',
            'geopy',
            'sympy',
        ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'Quantum network', 'Discrete-event simulation'],
        classifiers= [
            "Development Status :: 1 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)