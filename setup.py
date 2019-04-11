import io
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = 'v0.2.5'
# exec(open('rasa_nlu_gao/version.py').read())

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rasa-nlu-gao',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    version=__version__,
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
