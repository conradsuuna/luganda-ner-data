# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
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

# Lint as: python3
"""LugandaPII: PII for Luganda Language"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@InProceedings{huggingface:dataset,
title = {Luganda Ner Dataset},
author={many authors
},
year={2022}
}
"""

_DESCRIPTION = """\
LugandaPII is a named entity dataset consisting of PERSON, ORG, LOCATION, NORP, USERID and DATE entities.
The train/validation/test sets are available for the Luganda language.
"""

# for github, replace "tree" with "raw" for example;
# "https://github.com/conradsuuna/luganda-ner-data/tree/main/data" =>
# "https://github.com/conradsuuna/luganda-ner-data/raw/main/data/"
_URL = "https://github.com/conradsuuna/luganda-ner-data/raw/main/data"
_TRAINING_FILE = "train.txt"
_VAL_FILE = "val.txt"
_TEST_FILE = "test.txt"


class LugPIIConfig(datasets.BuilderConfig):
    """BuilderConfig for PII"""

    def __init__(self, **kwargs):
        """BuilderConfig for PII.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LugPIIConfig, self).__init__(**kwargs)


class PII(datasets.GeneratorBasedBuilder):
    """PII dataset."""

    BUILDER_CONFIGS = [
        LugPIIConfig(name="lug", version=datasets.Version("1.0.0"), description="PII NER Luganda dataset"),  
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PERSON",
                                "I-PERSON",
                                "L-PERSON",
                                "U-PERSON",
                                "B-NORP",
                                "I-NORP",
                                "L-NORP",
                                "U-NORP",
                                "B-DATE",
                                "I-DATE",
                                "L-DATE",
                                "U-DATE",
                                "B-USERID",
                                "I-USERID",
                                "L-USERID",
                                "U-USERID",
                                "B-ORG",
                                "I-ORG",
                                "L-ORG",
                                "U-ORG",
                                "B-LOCATION",
                                "I-LOCATION",
                                "L-LOCATION",
                                "U-LOCATION",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}/{_TRAINING_FILE}",
            "val": f"{_URL}/{_VAL_FILE}",
            "test": f"{_URL}/{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["val"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # since our tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
