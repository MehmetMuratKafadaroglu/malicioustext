from setuptools import setup
from setuptools.command.install import install

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        import nltk
        install.run(self)
        print("Downloading nltk")
        nltk.download('stopwords')
        nltk.download('omw-1.4')
        nltk.download('punkt')
        nltk.download('wordnet')
        
setup(
    name='offensivetext',
    version='0.2.0',    
    description='Filter out racism, sexism and sexual content with pretrained models',
    url='https://github.com/MehmetMuratKafadaroglu/offensivetext',
    author='Mehmet Kafadaroglu',
    author_email='mehmetkafadaroglu@gmail.com',
    license='BSD 2-clause',
    packages=['offensivetext'],
    install_requires=['sklearn',                    'scikit-learn',
                    'numpy', 
                    'nltk',                    'pandas',                    'scipy',                    'matplotlib',                      ],    package_data = {'': ['models/*.sav']},
    include_package_data = True,
    classifiers=[
       'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],    cmdclass={
        'install': PostInstallCommand,
    },
)