# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" The Movie Dialog dataset: evaluating models at goal and non-goal orientated dialog centered around the topic of movies."""

from __future__ import absolute_import, division, print_function

import re
import os

import datasets


_CITATION = """\
@misc{dodge2016evaluating,
      title={Evaluating Prerequisite Qualities for Learning End-to-End Dialog Systems}, 
      author={Jesse Dodge and Andreea Gane and Xiang Zhang and Antoine Bordes and Sumit Chopra and Alexander Miller and Arthur Szlam and Jason Weston},
      year={2016},
      eprint={1511.06931},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
The Movie Dialog dataset (MDD) from Facebook is designed to measure how well models can perform at goal and non-goal orientated dialog centered around the topic of movies (question answering, recommendation and discussion).
Details and baseline results on this dataset can be found in the paper.

The file format is the same as in the bAbI tasks. The IDs for a given dialog start at 1 and increase.
Each ID consists of one turn for each speaker (an “exchange”), which are tab separated.
When the IDs in a file reset back to 1 you can consider the following sentences as a new conversation.

We use the data provided by the ParlAI project at https://github.com/facebookresearch/ParlAI.
"""

_HOMEPAGE = "https://research.fb.com/downloads/babi/"

_URLs = {
    'TASK4_REDDIT': "http://cs.nyu.edu/~xiang/task4_reddit.tgz",
    'OTHER_TASKS': "http://parl.ai/downloads/moviedialog/moviedialog.tar.gz",
}

_FILE_PATH = {
    'qa': ['movie_dialog_dataset', 'task1_qa'],
    'recommendations': ['movie_dialog_dataset', 'task2_recs'],
    'qa_recommendations': ['movie_dialog_dataset', 'task3_qarecs'],
    'reddit': ['task4_reddit'],
}

_SPLIT_TO_SUFFIX = {
    datasets.Split.TRAIN: '_train.txt',
    datasets.Split.VALIDATION: '_dev.txt',
    datasets.Split.TEST: '_test.txt',
}

_LICENSE = """CC License"""

class MovieDialogFb(datasets.GeneratorBasedBuilder):
    """ The Movie Dialog dataset: evaluating models at goal and non-goal orientated dialog centered around the topic of movies."""

    VERSION = datasets.Version("3.01")  # Version of ParlAI for easy data source comparison

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="qa", description="Closed-domain QA dataset asking templated questions about movies, answerable from Wikipedia, similar to WikiMovies."),
        datasets.BuilderConfig(name="recommendations", description="Questions asking for movie recommendations."),
        datasets.BuilderConfig(name="qa_recommendations", description="Dialogs discussing questions about movies as well as recommendations."),
        datasets.BuilderConfig(name="reddit", description="Dialogs discussing Movies from Reddit (the Movies SubReddit)."),
        datasets.BuilderConfig(name="knowledge_base", description="The knowledge base of information about the movies, actors and other entities that are mentioned in the dialogs."),
    ]

    # DEFAULT_CONFIG_NAME = "qa"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method pecifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features = datasets.Features(
                {
                    "conversation_id": datasets.Value("int32"),
                    "utterance_id": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive 
        if self.config.name == "reddit":
            my_urls = _URLs["TASK4_REDDIT"]
        else:
            my_urls = _URLs["OTHER_TASKS"]
        data_dir = dl_manager.download_and_extract(my_urls)
        dpath = os.path.join(data_dir, *(_FILE_PATH[self.config.name]))
        return [
            datasets.SplitGenerator(
                name=split,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, _FILE_PATH[self.config.name][-1] + _SPLIT_TO_SUFFIX[split]),
                },
            ) for split in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        ]

    def _generate_examples(self, filepath):
        """ Yields examples. """
        # TODO: This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)

        """ Yields examples. """
        pattern = re.compile(r"(\d+)\s([^\t]+)\t?([^\t]*)?")
        with open(filepath) as f:
            data = f.read().splitlines()
            conversation_id = 0
            utterance_id = 0
            for id_, row in enumerate(data):
                m = pattern.match(row)
                new_id = int(m.group(1))
                if new_id < utterance_id:
                    conversation_id += 1

                utterance_id = new_id

                if m.group(3):
                    answers = m.group(3).replace(',', ' ')
                else:
                    answers = m.group(3)

                yield id_, {
                    "utterance_id": utterance_id,
                    "text": m.group(2),
                    "answer": answers,
                    "conversation_id": conversation_id,
                }
