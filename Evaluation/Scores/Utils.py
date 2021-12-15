import deprecate

from const import ISSUE_SPECIFIC_FRAME_END_TOKEN, GENERIC_MAPPED_FRAME_END_TOKEN, GENERIC_INFERRED_FRAME_END_TOKEN, \
    TOPIC_END_TOKEN
from MainUtils import clean_premise as main_clean_premise
from loguru import logger


@deprecate.deprecated(target=main_clean_premise)
def clean_premise(premise: str) -> str:
    logger.trace("OK, you want to remove the control code from the premise: \"{}\"", premise)
    for tok in [ISSUE_SPECIFIC_FRAME_END_TOKEN, GENERIC_MAPPED_FRAME_END_TOKEN, GENERIC_INFERRED_FRAME_END_TOKEN]:
        if tok in premise:
            index = premise.index(tok)
            logger.trace("Found the frame end token \"{}\" at position {} - cut", tok, index)
            premise = premise[index + len(tok):]
    if TOPIC_END_TOKEN in premise:
        index = premise.index(TOPIC_END_TOKEN)
        logger.trace("Found the topic end token \"{}\" at position {} - cut", TOPIC_END_TOKEN, index)
        premise = premise[index + len(TOPIC_END_TOKEN):]

    logger.trace("Last cleaning steps...")
    premise = premise.strip(" :")

    return premise
