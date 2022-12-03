from setuptools import setup

setup(
    name='malicioustext',
    version='0.1.0',    
    description='Python package that finds malicious text',
    url='https://github.com/MehmetMuratKafadaroglu/malicioustext',
    author='Mehmet Kafadaroglu',
    author_email='mehmetkafadaroglu@gmail.com',
    license='BSD 2-clause',
    packages=['malicioustext'],
    install_requires=['sklearn',
                    'numpy', 
                    'nltk',
                      ],

    classifiers=[
       'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)