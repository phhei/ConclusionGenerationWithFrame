import math
import pathlib
import re
from typing import Union, Optional, List, Tuple, Dict

import nltk
import numpy
import pandas
import torch
from loguru import logger

if pathlib.Path.home().name == "Philipp":
    glove_path = pathlib.Path.home().joinpath("Documents", "Einsortiertes", "Nach Lebensabschnitten einsortiert",
                                              "Promotionszeit (2019-2023)", "Promotion", "Programming",
                                              "_wordEmbeddings", "glove", "glove.840B.300d.txt")
elif pathlib.Path.home().name == "pheinisch":
    # ssh compute
    glove_path = pathlib.Path("..", "glove", "glove.840B.300d.txt")
else:
    glove_path = None

stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "-", ">", "->"}


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

        self.add_other = add_other

        logger.info("Now load the word embeddings from \"{}\" - this may take a while...", glove_path.name)
        if not glove_path.exists() or not glove_path.is_file():
            logger.critical("The core, the word embeddings, are not loadable: {}", glove_path.absolute())
            exit(-42)

        self.w2v = dict()
        with glove_path.open(mode="r", encoding="utf-8") as glove_reader:
            for line in glove_reader:
                splits = line.split(" ")
                # logger.trace("Read word \"{}\" ({}d)", splits[0].strip(), len(splits[1:]))
                self.w2v[splits[0].strip()] = numpy.fromiter(iter=splits[1:], dtype=numpy.float)
        logger.success("Read successfully {} words from \"{}\"", len(self.w2v), glove_path.name)

        self.data.insert(loc=2, column="w2v_keywords_label",
                         value=[numpy.average(
                             a=[self.w2v.get(s, numpy.zeros_like(self.w2v["the"]))for s in c.split(" ")], axis=0
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
                data=[("ecology", "nature ecology",
                       numpy.average([self.w2v.get("nature", numpy.zeros_like(self.w2v["the"])),
                                      self.w2v.get("ecology", numpy.zeros_like(self.w2v["the"]))], axis=0),
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
        if issue_specific_frame.upper().strip() == "OTHER":
            logger.debug("The issue_specific_frame matches exactly the \"{}\" frame - "
                         "hence, skip all the work and but it into the other-bucket :)", issue_specific_frame)
            return self.data.index.values.max() + 1 if fetch_column is None else "other"

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
            [self.w2v.get(s, numpy.zeros_like(self.w2v["the"])) for s in issue_specific_frame_splits]
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
        if o_l[0][1] == numpy.nan or (o_l[0][1] <= threshold_accept and abs(o_l[0][1]-o_l[-1][1]) <= 1/3):
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
            logger.opt(exception=False).error("You want to retrieve the rank \"{}\", but you have only {} frames in {}",
                                              rank, len(self.data), self.name)
            return "other" if self.add_other else "n/a"

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
                    logger.warning("Only strings and ints are allowed as frame-id-input (first element of each "
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
