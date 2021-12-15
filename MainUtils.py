from pathlib import Path

from loguru import logger
from typing import Optional, Dict
from gensim.models.keyedvectors import KeyedVectors
from gensim.downloader import load as w2v_load

import const

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


def retrieve_bracket_substring(message: str, bracket_start: str, bracket_end: str) -> Optional[str]:
    logger.trace("OK, let's extract the \"{} STRING {}\" out of \"{}\"", bracket_start, bracket_end, message)
    try:
        return message[message.index(bracket_start)+len(bracket_start):message.index(bracket_end)].strip()
    except ValueError:
        logger.opt(exception=False).debug("The string \"{}\" doesn't contain \"{}\" / \"{}\"", message, bracket_start,
                                          bracket_end)
        return None


def retrieve_issue_specific_frame(premise: str, default: str = "other") -> str:
    ret = retrieve_bracket_substring(message=premise,
                                     bracket_start=const.ISSUE_SPECIFIC_FRAME_START_TOKEN,
                                     bracket_end=const.ISSUE_SPECIFIC_FRAME_END_TOKEN)

    if ret is None:
        logger.info("The premise \"{}\" doesn't contain any information about the issue-specific frame (return \"{}\")",
                    premise, default)
        return default
    else:
        logger.trace("Retrieved successfully the issue_specific-frame: \"{}\"", ret)
        return ret


def retrieve_generic_frame(premise: str, default: str = "other") -> str:
    ret = retrieve_bracket_substring(message=premise,
                                     bracket_start=const.GENERIC_MAPPED_FRAME_START_TOKEN,
                                     bracket_end=const.GENERIC_MAPPED_FRAME_END_TOKEN)

    if ret is None:
        logger.debug("The premise \"{}\" doesn't contain any information about the mapped (issue_specific -> generic) "
                     "generic frame", premise)
        ret = retrieve_bracket_substring(message=premise,
                                         bracket_start=const.GENERIC_INFERRED_FRAME_START_TOKEN,
                                         bracket_end=const.GENERIC_INFERRED_FRAME_END_TOKEN)
        if ret is None:
            logger.warning("There is no information about the used generic frame in \"{}\", "
                           "let's stick to the default \"{}\"", premise, default)
            return default
        else:
            return ret
    else:
        logger.trace("Perfect, the (mapped) generic frame is found: \"{}\"", ret)
        return ret


def retrieve_frame(premise: str) -> str:
    default = "<{}>".format(premise[len("summarize"):min(len(premise), len("summarize")+8)].strip(" :")
                            if premise.startswith("summarize") else premise[:min(8, len(premise))])

    ret = retrieve_issue_specific_frame(premise=premise, default=default)

    if ret == default:
        ret = retrieve_generic_frame(premise=premise, default=default)

    logger.debug("Retrieved the frame of \"{}\": \"{}\"", premise, ret)

    return ret


def retrieve_topic(premise: str) -> str:
    ret = retrieve_bracket_substring(message=premise,
                                     bracket_start=const.TOPIC_START_TOKEN,
                                     bracket_end=const.TOPIC_END_TOKEN)

    if ret is None:
        logger.debug("The string \"{}\" doesn't reveal anything about the topic :\\", premise)
        return premise

    return ret


loaded_glove_w2v_models: Dict[str, KeyedVectors] = dict()


def get_glove_w2v_model(path: Optional[Path] = None) -> KeyedVectors:
    if path is None:
        if Path.home().name == "Philipp":
            glove_path = Path.home().joinpath("Documents", "Einsortiertes", "Nach Lebensabschnitten einsortiert",
                                              "Promotionszeit (2019-2023)", "Promotion", "Programming",
                                              "_wordEmbeddings", "glove", "glove.840B.300d.txt")
            logger.info("Glove-local: {}", glove_path)
        elif Path.home().name == "pheinisch":
            # ssh compute
            glove_path = Path("..", "glove", "glove.840B.300d.txt")
            logger.info("Glove-ssh: {}", glove_path)
        else:
            logger.warning("Can't infer the path in which the glove files are...")
            glove_path = None
    else:
        logger.info("Follow thw glove-path... {}", path.absolute())
        if path.exists():
            glove_path = path
        else:
            logger.warning("Your given path ({}) doesn't exist!", path)
            glove_path = None

    key = "glove-wiki-gigaword-300" if glove_path is None else str(glove_path.absolute())
    if key in loaded_glove_w2v_models:
        logger.success("\"{}\" is already loaded: {}", key, loaded_glove_w2v_models[key])
        return loaded_glove_w2v_models[key]
    else:
        logger.warning("We have to load \"{}\" first!", key)
        if glove_path is None:
            loaded_glove_w2v_models[key] = w2v_load(key)
        else:
            loaded_glove_w2v_models[key] = KeyedVectors.load_word2vec_format(key, binary=False, no_header=True)

        picked_model = loaded_glove_w2v_models[key]
        picked_model.fill_norms(force=False)
        logger.success("Loaded the model \"{}\" ({}x{}d)",
                       key, len(picked_model.key_to_index), picked_model.vector_size)

        return picked_model


def clean_premise(premise: str) -> str:
    logger.trace("OK, you want to remove the control code from the premise: \"{}\"", premise)
    for tok in [const.ISSUE_SPECIFIC_FRAME_END_TOKEN, const.GENERIC_MAPPED_FRAME_END_TOKEN,
                const.GENERIC_INFERRED_FRAME_END_TOKEN]:
        if tok in premise:
            index = premise.index(tok)
            logger.trace("Found the frame end token \"{}\" at position {} - cut", tok, index)
            premise = premise[index + len(tok):]
    if const.TOPIC_END_TOKEN in premise:
        index = premise.index(const.TOPIC_END_TOKEN)
        logger.trace("Found the topic end token \"{}\" at position {} - cut", const.TOPIC_END_TOKEN, index)
        premise = premise[index + len(const.TOPIC_END_TOKEN):]

    logger.trace("Last cleaning steps...")
    premise = premise.strip(" :")

    return premise
