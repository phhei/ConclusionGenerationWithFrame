from const import FRAME_END_TOKEN, TOPIC_END_TOKEN
from loguru import logger


def clean_premise(premise: str) -> str:
    logger.trace("OK, you want to remove the control code from the premise: \"{}\"", premise)
    if FRAME_END_TOKEN in premise:
        index = premise.index(FRAME_END_TOKEN)
        logger.trace("Found the frame end token \"{}\" at position {} - cut", FRAME_END_TOKEN, index)
        premise = premise[index + len(FRAME_END_TOKEN):]
    if TOPIC_END_TOKEN in premise:
        index = premise.index(TOPIC_END_TOKEN)
        logger.trace("Found the topic end token \"{}\" at position {} - cut", TOPIC_END_TOKEN, index)
        premise = premise[index + len(TOPIC_END_TOKEN):]

    logger.trace("Last cleaning steps...")
    premise = premise.strip(" :")

    return premise
