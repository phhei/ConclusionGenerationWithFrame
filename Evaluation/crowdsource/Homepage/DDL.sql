CREATE TABLE CrowdSourceArgument(
    argument_ID VARCHAR(100) PRIMARY KEY,
    topic TINYTEXT NOT NULL,
    premise TEXT NOT NULL,
    issue_specific_frame TINYTEXT NOT NULL,
    generic_mapped_frame TINYTEXT NOT NULL,
    generic_inferred_frame TINYTEXT
);

CREATE TABLE CrowdSourceConclusion(
    argument_ID VARCHAR(100) NOT NULL,
    conclusion_identifier VARCHAR(100) NOT NULL,
    conclusion_text TEXT NOT NULL,
    order_number int,
    round int,
    PRIMARY KEY (argument_ID, conclusion_identifier),
    FOREIGN KEY (argument_ID) REFERENCES CrowdSourceArgument(argument_ID) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE CrowdSourceAnswer(
    a_ID int NOT NULL PRIMARY KEY AUTO_INCREMENT,
    annotator_ID int NOT NULL,
    argument_ID VARCHAR(100) NOT NULL,
    conclusion_identifier_1 VARCHAR(100) NOT NULL,
    conclusion_identifier_2 VARCHAR(100),
    timeInS int NOT NULL,
    a_validity int NOT NULL,
    a_novelty int NOT NULL,
    a_issue_specific_frame int,
    a_generic_mapped_frame int,
    a_generic_inferred_frame int,
    a_comment TEXT,
    FOREIGN KEY (argument_ID, conclusion_identifier_1) REFERENCES CrowdSourceConclusion(argument_ID, conclusion_identifier) ON UPDATE CASCADE ON DELETE CASCADE
);