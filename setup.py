import io
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = 'v0.1.7'
# exec(open('rasa_nlu_gao/version.py').read())

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

tests_requires = [
    "pytest",
    "pytest-pep8",
    "pytest-services",
    "pytest-cov",
    "pytest-twisted<1.6",
    "treq"
]

install_requires = [
    "pathlib",
    "cloudpickle",
    "gevent",
    "klein",
    "boto3",
    "packaging",
    "typing",
    "future",
    "six",
    "tqdm",
    "requests",
    "jsonschema",
    "matplotlib",
    "numpy>=1.13",
    "simplejson",
    "pyyaml",
    'coloredlogs',
]

extras_requires = {
    'test': tests_requires,
    'spacy': ["scikit-learn",
              "sklearn-crfsuite",
              "scipy",
              "spacy>2.0",
              ],
    'tensorflow': ["scikit-learn",
                   "sklearn-crfsuite",
                   "scipy",
                   "tensorflow"
                   ],
    'mitie': ["mitie"],
}

setup(
    name='rasa-nlu-gao',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        # supported python versions
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
    ],
    version=__version__,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
    include_package_data=True,
    description="Rasa NLU a natural language parser for bots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gao Quan',
    author_email='gaoquan199035@gmail.com',
    maintainer="Gao Quan",
    maintainer_email="gaoquan199035@gmail.com",
    license='Apache 2.0',
    url="https://rasa.com",
    keywords="nlp machine-learning machine-learning-library bot bots "
             "botkit rasa conversational-agents conversational-ai chatbot"
             "chatbot-framework bot-framework",
    download_url="https://github.com/GaoQ1/rasa_nlu_gq/archive/{}.tar.gz"
                 "".format(__version__),
    project_urls={
        'Bug Reports': 'https://github.com/GaoQ1/rasa_nlu_gq/issues',
        'Source': 'https://github.com/GaoQ1/rasa_nlu_gq',
    },
)

print("\nWelcome to Rasa NLU!")
print("If any questions please visit documentation "
      "page https://nlu.rasa.com")
