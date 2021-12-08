import math
import pathlib
import random
import re
from typing import Union, Optional, List, Tuple, Dict

import nltk
import numpy
import pandas
import torch
from loguru import logger

from MainUtils import get_glove_w2v_model
from MainUtils import stop_words


class FrameSet:
    def __init__(self, frame_set: Union[str, pathlib.Path], add_other: bool = True) -> None:
        logger.debug("Selected following frame set: {}", frame_set)

        if isinstance(frame_set, str):
            frame_set_str = frame_set
            frame_set = pathlib.Path("frame_sets", "{}.csv".format(frame_set))
            logger.info("Redirect \"{}\" to -> {}", frame_set_str, frame_set)

        if not frame_set.exists() or not frame_set.is_file():
            logger.error("The frame set \"{}\" doesn't exists (as file)!", frame_set.absolute())
            self.name = "Not loaded"
            self.data = pandas.DataFrame(columns=["ID", "label", "keywords_label", "description"])
            add_other = True
        else:
            self.name = frame_set.stem
            self.data = pandas.read_csv(filepath_or_buffer=str(frame_set.absolute()), verbose=True, index_col=0)
            logger.success("Successfully loaded {} frames{}.", len(self.data), " (+ OTHER frame)" if add_other else "")
            frame_mapping_path = frame_set.parent.joinpath("{}.map".format(frame_set.stem))
            if frame_mapping_path.exists():
                try:
                    self.predefined_mappings = {
                        k: v["ID"] for k, v in pandas.read_csv(filepath_or_buffer=str(frame_mapping_path.absolute()),
                                                               verbose=False, index_col=0, header=None,
                                                               names=["String", "ID"],
                                                               dtype={"String": "str", "ID": "int"}
                                                               ).to_dict(orient="index").items()
                    }
                    logger.success("Successfully load {} string->frame-ID hints from \"{}\"",
                                   len(self.predefined_mappings), frame_mapping_path.name)
                except IndexError:
                    logger.opt(exception=False).error("Your mapping file \"{}\" is malformed. Expected two values per "
                                                      "line (\"String\", \"Frame-ID\") but got more values!",
                                                      frame_mapping_path.name)
                    self.predefined_mappings = dict()
                except ValueError:
                    logger.opt(exception=True).error("Only integer are allowed (frame-IDs) in the second column."
                                                     "Please check your mapping-file \"{}\"", frame_mapping_path.name)
                    self.predefined_mappings = dict()
            else:
                logger.info("No String-FrameID-Mapping-file defined here: {}", frame_mapping_path)
                self.predefined_mappings = dict()

        self.add_other = add_other

        self.w2v_model = get_glove_w2v_model()
        self.data.insert(loc=2, column="w2v_keywords_label",
                         value=[numpy.average(
                             a=[self.w2v_model.get_vector(s, norm=True)
                                if s in self.w2v_model else
                                numpy.zeros((self.w2v_model.vector_size,), dtype=self.w2v_model.vectors.dtype)
                                for s in c.split(" ")], axis=0
                         ) for c in self.data["keywords_label"]])
        logger.debug(self.data)

    def __str__(self) -> str:
        return "{} ({} frames)".format(self.name, len(self.data)+int(self.add_other))

    def __hash__(self) -> int:
        return hash(self.name)

    def __len__(self) -> int:
        return len(self.data)+int(self.add_other)

    def add_ecology_frame(self):
        self.data = self.data.append(
            pandas.DataFrame.from_records(
                data=[("ecology", "ecology environmental sustainability",
                       numpy.average([self.w2v_model.get_vector(s, norm=True)
                                      if s in self.w2v_model else
                                      numpy.zeros((self.w2v_model.vector_size,), dtype=self.w2v_model.vectors.dtype)
                                      for s in ["nature", "ecology"]],
                                     axis=0),
                       "ecological aspect, effects on the environment")],
                columns=self.data.columns
            ),
            ignore_index=True
        )
        self.name += " + ecology"
        logger.success("Added successfully the frame. Now you have {}.", len(self.data))

    def issues_specific_frame_to_generic(self, issue_specific_frame: str, topic: Optional[str] = None,
                                         rank: int = 0,
                                         fetch_column: Optional[str] = "keywords_label",
                                         semantic_reordering: bool = True,
                                         remove_stopwords: bool = True) -> Union[int, str]:
        try:
            if self.add_other and issue_specific_frame.upper().strip() == "OTHER":
                logger.debug("The issue_specific_frame matches exactly the \"{}\" frame - "
                             "hence, skip all the work and but it into the other-bucket :)",
                             issue_specific_frame.strip())
                return self.data.index.values.max() + 1 if fetch_column is None else "other"

            if rank == 0 and issue_specific_frame in self.predefined_mappings:
                idx = self.predefined_mappings[issue_specific_frame]
                logger.debug("\"{}\" is already predefined: {}", issue_specific_frame, idx)
                if fetch_column is None:
                    if idx >= 0:
                        return idx
                    elif self.add_other:
                        return self.data.index.values.max()+1
                    else:
                        logger.error("You want to use the \"other\"-frame ({}) in {} without setting the other-bucket. "
                                     "Select a random class...", idx, self)
                        return random.randint(0, len(self)-1)
                else:
                    try:
                        if idx < 0 and self.add_other:
                            return "other"
                        return self.data[fetch_column][self.predefined_mappings[issue_specific_frame]]
                    except IndexError:
                        logger.opt(exception=False).warning("Index \"{}\" is not available, but in occur in the "
                                                            "predefined mappings ({})",
                                                            self.predefined_mappings[issue_specific_frame],
                                                            issue_specific_frame)
                        if self.add_other:
                            logger.trace("OK, let's put it into the \"other\"-bucket")
                            return "other"
                        else:
                            logger.critical("No \"other\"-frame available - we don't know how to treat the entry: {}: {}",
                                            self.predefined_mappings[issue_specific_frame],
                                            issue_specific_frame)
                            return "n/a"

            issue_specific_frame_splits = [s.lower().strip(" .\"'")
                                           for s in re.split(pattern="\s+|-|/", string=issue_specific_frame,
                                                             maxsplit=15 if topic is None else 6)
                                           if len(s) >= 1]
            logger.debug("Retrieved a issue-specific-frame: \"{}\" -> {}", issue_specific_frame,
                         issue_specific_frame_splits)

            if semantic_reordering and len(issue_specific_frame_splits) >= 2:
                def semantic_reorder(tokens:List[str]) -> List[str]:
                    tokens_pos = nltk.pos_tag(tokens=tokens, tagset="universal")
                    noun_tokens = [tok for tok, pos in tokens_pos if pos == "NOUN"]
                    noun_tokens.reverse()
                    non_noun_tokens = [tok for tok, pos in tokens_pos if pos != "NOUN"]
                    return noun_tokens + non_noun_tokens
                try:
                    issue_specific_frame_splits = semantic_reorder(issue_specific_frame_splits)
                    logger.trace("Reordered the frame splits: {}", " > ".join(issue_specific_frame_splits))
                except LookupError:
                    logger.warning("There is a nltk-module missing - we execute "
                                   "\"nltk.download('averaged_perceptron_tagger')\" first!")
                    if nltk.download("averaged_perceptron_tagger") and nltk.download("universal_tagset"):
                        issue_specific_frame_splits = semantic_reorder(issue_specific_frame_splits)
                        logger.success("Success - reordered the frame splits: {}", " > ".join(issue_specific_frame_splits))
                    else:
                        logger.critical("We're not able to install the necessary nltk-module. Hence, we must disable the "
                                        "semantic reordering")
                        semantic_reordering = False

            if remove_stopwords:
                logger.trace("Let's remove the stopwords from the {} tokens", len(issue_specific_frame_splits))
                issue_specific_frame_splits = [s for s in issue_specific_frame_splits if s not in stop_words]

            logger.debug("Final words to weight: {}", "-".join(issue_specific_frame_splits))
            if len(issue_specific_frame_splits) == 0:
                logger.warning("The frame \"{}\" results in an empty one after preprocessing - let's sort into \"other\"")
                issue_specific_frame_splits = ["other"]

            issue_specific_frame_vecs =\
                [self.w2v_model.get_vector(s, norm=True)
                 if s in self.w2v_model else
                 numpy.zeros((self.w2v_model.vector_size,), dtype=self.w2v_model.vectors.dtype)
                 for s in issue_specific_frame_splits]
            logger.trace("Collected {} numpy-word-embeddings", len(issue_specific_frame_vecs))

            if semantic_reordering:
                issue_specific_frame_vecs = [math.exp(-i)*v for i, v in enumerate(issue_specific_frame_vecs)]

            issue_specific_frame_vec =\
                numpy.average(issue_specific_frame_vecs, axis=0,
                              weights=numpy.exp2(numpy.flip(numpy.arange(start=1,
                                                                         stop=len(issue_specific_frame_vecs)+1),
                                                            axis=0)) if semantic_reordering else None)
            logger.trace("issue_specific_frame_vec: {}", issue_specific_frame_vec)

            def cos_sim(a: numpy.ndarray, b: numpy.ndarray):
                return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

            o_l = [(i, cos_sim(issue_specific_frame_vec, self.data["w2v_keywords_label"][i])) for i in self.data.index]
            o_l.sort(key=lambda k: k[1], reverse=True)
            logger.debug("Sorted the similarities: {}", o_l)
            threshold_accept = 5/8 if topic is None else 2/3
            required_gap = 1/3 * (2/3 + 1/3 * (1+math.log(len(self)/15, 10)))
            if o_l[0][1] == numpy.nan or (o_l[0][1] <= threshold_accept and abs(o_l[0][1]-o_l[-1][1]) <= required_gap):
                logger.info("No frame fits in a good manner to \"{}\". The closest frame is {} with {}",
                            issue_specific_frame, o_l[0][0], round(o_l[0][1], 2))
                if topic is not None:
                    logger.info("You insert also the topic \"{}\" - maybe this will help to define \"{}\" "
                                "in a better way.", topic, issue_specific_frame)
                    return self.issues_specific_frame_to_generic(
                        issue_specific_frame="{} -> {}".format(topic, issue_specific_frame),
                        topic=None, rank=rank, fetch_column=fetch_column,
                        semantic_reordering=True, remove_stopwords=remove_stopwords
                    )

                if self.add_other and rank == 0:
                    logger.trace("OK, let's put it into the \"other\"-bucket")
                    return self.data.index.values.max()+1 if fetch_column is None else "other"

            try:
                if fetch_column is None:
                    return o_l[rank][0]
                elif fetch_column in self.data.columns:
                    return self.data[fetch_column][o_l[rank][0]]
                else:
                    logger.warning("The dataframe doesn't contain the desired column \"{}\" - only {}", fetch_column,
                                   self.data.columns)
                    return self.data["keywords_label"][o_l[rank][0]]
            except IndexError:
                logger.opt(exception=False).error("You want to retrieve the rank \"{}\", "
                                                  "but you have only {} frames in {}",
                                                  rank, len(self.data), self.name)
                return "other" if self.add_other else "n/a"
        except AttributeError:
            logger.opt(exception=True).error("Received a wrong (typed) parameter. "
                                             "issue_specific_frame must be a string but is \"{}\" ({}). "
                                             "Same holds for topic (but is optional)", issue_specific_frame,
                                             type(issue_specific_frame))

    def get_frame_count_dict(self, corpora: List[List[Tuple[Union[int, str], torch.Tensor]]], vocab_size: int)\
            -> Dict[int, torch.Tensor]:
        logger.debug("OK, let's process {} corpora "
                     "(Frame -> tokenized vec-index-mapped text which reflects that frame)", len(corpora))
        counter_dict = {i: torch.zeros(vocab_size, dtype=torch.int64) for i in self.data.index}
        if self.add_other:
            logger.trace("Must add the other class, too (extend keys [{}] with one additional)",
                         " ".join(map(lambda k: str(k), counter_dict.keys())))
            counter_dict[max(counter_dict.keys())+1] = torch.zeros(vocab_size, dtype=torch.int64)

        for corpus in corpora:
            logger.trace("Let's process the (next) corpora with {} samples", len(corpus))
            for sample_frame, sample_text in corpus:
                if isinstance(sample_frame, str):
                    logger.trace("The frame is decoded as text ({}), let's convert it", sample_frame)
                    sample_frame = self.issues_specific_frame_to_generic(issue_specific_frame=sample_frame,
                                                                         semantic_reordering=False,
                                                                         fetch_column=None)
                elif not isinstance(sample_frame, int):
                    logger.warning("Only strings and ints are allowed as frame-sid-input (first element of each "
                                   "corpus-tuple), but you input a {}!", type(sample_frame))
                    if isinstance(sample_frame, torch.Tensor):
                        sample_frame = sample_frame.item()
                
                for token in torch.unique(sample_text):
                    counter_dict[sample_frame][token.item()] += 1
                logger.trace("Processed the sample {} -> {}", sample_frame, sample_text)

        sum_maples = sum(map(lambda c: len(c), corpora))
        counter_dict[-1] = torch.full((vocab_size,), fill_value=sum_maples, dtype=torch.int64)
        logger.success("Finished the count-dict with {} frame-entries and {} processed samples", len(counter_dict),
                       sum_maples)

        return counter_dict
