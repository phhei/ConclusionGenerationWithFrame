CREATE TABLE CrowdSourceSamples(
    test_ID int PRIMARY KEY AUTO_INCREMENT,
    topic TINYTEXT NOT NULL,
    premise TEXT NOT NULL,
    conclusion1ID TINYTEXT NOT NULL,
    conclusion1 TEXT NOT NULL,
    conclusion2ID TINYTEXT NOT NULL,
    conclusion2 TEXT NOT NULL,
    issuespecificframe TINYTEXT,
    genericmappedframe TINYTEXT,
    genericinferredframe TINYTEXT
);

CREATE TABLE CrowdSourceAnswer(
    a_ID int PRIMARY KEY AUTO_INCREMENT,
    annotator_ID int NOT NULL,
    test_ID int,
    timeInS int NOT NULL,
    a_validity int NOT NULL,
    a_novelty int NOT NULL,
    a_issuespecificframe int,
    a_genericmappedframe int,
    a_genericinferredframe int,
    a_comment TEXT,
    FOREIGN KEY (test_ID) REFERENCES CrowdSourceSamples(test_ID) ON UPDATE CASCADE ON DELETE SET NULL
);