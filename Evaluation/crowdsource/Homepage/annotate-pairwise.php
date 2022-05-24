<?php
    require "password.php";

    $annotation_round = ($_GET and array_key_exists("additional_args", $_GET)) ? "(101)" : "(100, 101)";

    if($_POST and array_key_exists("annotator_ID", $_GET)) {
        $conn_db = new mysqli("localhost", "philipp", $db_password, "philipp_umfrage");
        $stat = $conn_db->error . '<br>'. $conn_db->stat() .'<br>';

        $annotator_ID = $conn_db->real_escape_string($_GET["annotator_ID"]);
        $sample_ID_commit = $conn_db->real_escape_string($_POST["sample_ID"]);
        $conclusion1_ID_commit = $conn_db->real_escape_string($_POST["conclusion1_id"]);
        $conclusion2_ID_commit = $conn_db->real_escape_string($_POST["conclusion2_id"]);

        # Conclusion  1
        if ($conn_db->query("SELECT COUNT(*) FROM CrowdSourceAnswer WHERE annotator_ID = ". $annotator_ID ." and argument_ID = '". $sample_ID_commit . "' and conclusion_identifier_1 = '". $conclusion1_ID_commit ."' and conclusion_identifier_2 IS NULL;")->fetch_array(MYSQLI_NUM)[0] >= 1) {
           ;
        } else {
            $validity = $conn_db->real_escape_string($_POST["c1_validity"]);
            $novelty = $conn_db->real_escape_string($_POST["c1_novelty"]);
            $generalFraming = array_key_exists("c1_generalFraming", $_POST) ? $conn_db->real_escape_string($_POST["c1_generalFraming"]) : "NULL";
            $specificFraming = array_key_exists("c1_specificFraming", $_POST) ? $conn_db->real_escape_string($_POST["c1_specificFraming"]) : "NULL";
            $time = time() - $_POST["timeStart"];
            $comments = $conn_db->real_escape_string($_POST["comments"]);
            $comments = (is_null($comments) or $comments == "") ? "no comment" : $comments;
            $query = "INSERT INTO CrowdSourceAnswer (annotator_ID, argument_ID, conclusion_identifier_1, timeInS, a_validity, a_novelty, a_issue_specific_frame, a_generic_mapped_frame, a_comment) VALUES
                                            (".$annotator_ID.", \"". $sample_ID_commit ."\", \"". $conclusion1_ID_commit ."\", ". $time .", ".$validity .", ".$novelty.", ".$specificFraming.", ".$generalFraming.", \"".$comments."\");";
            if($conn_db->query($query) === false) {
                $error_msg = " WARNING: Your last submission failed unfortunately (". $conn_db->error .")";
            } else {
                $error_msg = null;
            }
        }
        
        # Conclusion 2
        if ($conn_db->query("SELECT COUNT(*) FROM CrowdSourceAnswer WHERE annotator_ID = ". $annotator_ID ." and argument_ID = '". $sample_ID_commit . "' and conclusion_identifier_1 = '". $conclusion2_ID_commit ."' and conclusion_identifier_2 IS NULL;")->fetch_array(MYSQLI_NUM)[0] >= 1) {
            ;
        } else {
            $validity = $conn_db->real_escape_string($_POST["c2_validity"]);
            $novelty = $conn_db->real_escape_string($_POST["c2_novelty"]);
            $generalFraming = array_key_exists("c2_generalFraming", $_POST) ? $conn_db->real_escape_string($_POST["c2_generalFraming"]) : "NULL";
            $specificFraming = array_key_exists("c2_specificFraming", $_POST) ? $conn_db->real_escape_string($_POST["c2_specificFraming"]) : "NULL";
            $time = time() - $_POST["timeStart"];
            $comments = $conn_db->real_escape_string($_POST["comments"]);
            $comments = (is_null($comments) or $comments == "") ? "no comment" : $comments;
            $query = "INSERT INTO CrowdSourceAnswer (annotator_ID, argument_ID, conclusion_identifier_1, timeInS, a_validity, a_novelty, a_issue_specific_frame, a_generic_mapped_frame, a_comment) VALUES
                                             (".$annotator_ID.", \"". $sample_ID_commit ."\", \"". $conclusion2_ID_commit ."\", ". $time .", ".$validity .", ".$novelty.", ".$specificFraming.", ".$generalFraming.", \"".$comments."\");";
            if($conn_db->query($query) === false) {
                $error_msg = " WARNING: Your last submission failed unfortunately (". $conn_db->error .")";
            } else {
                    $error_msg = null;
            }
        }
        # both conclusions compared
        if ($conn_db->query("SELECT COUNT(*) FROM CrowdSourceAnswer WHERE annotator_ID = ". $annotator_ID ." and argument_ID = '". $sample_ID_commit . "' and conclusion_identifier_1 = '". $conclusion1_ID_commit ."' and conclusion_identifier_2 = '". $conclusion2_ID_commit ."';")->fetch_array(MYSQLI_NUM)[0] >= 1) {
            $error_msg = "You already annotated this pairwise sample!";
        } else {
            $validity = $conn_db->real_escape_string($_POST["c2c_validity"]);
            $novelty = $conn_db->real_escape_string($_POST["c2c_novelty"]);
            $generalFraming = array_key_exists("c2c_generalFraming", $_POST) ? $conn_db->real_escape_string($_POST["c2c_generalFraming"]) : "NULL";
            $specificFraming = array_key_exists("c2c_specificFraming", $_POST) ? $conn_db->real_escape_string($_POST["c2c_specificFraming"]) : "NULL";
            $time = time() - $_POST["timeStart"];
            $comments = $conn_db->real_escape_string($_POST["comments"]);
            $comments = (is_null($comments) or $comments == "") ? "no comment" : $comments;

            $query = "INSERT INTO CrowdSourceAnswer (annotator_ID, argument_ID, conclusion_identifier_1, conclusion_identifier_2, timeInS, a_validity, a_novelty, a_issue_specific_frame, a_generic_mapped_frame, a_comment) VALUES
                                            (".$annotator_ID.", \"".$sample_ID_commit."\", \"".$conclusion1_ID_commit."\", \"".$conclusion2_ID_commit."\", ". $time .", ".$validity .", ".$novelty.", ".$specificFraming.", ".$generalFraming.", \"".$comments."\");";

            if($conn_db->query($query) === FALSE) {
                $error_msg = " WARNING: Your last submission failed unfortunately (". $conn_db->error .")";
            } else {
                $error_msg = null;
            }
        }
        $conn_db->close();
    }

    if(array_key_exists("annotator_ID", $_GET) === FALSE) {
        $sample_ID = "n/a";
        $samples_done = 0;
        $samples_total = 1;
        $topic = "No annotator-ID provided!";
        $premise = "Please use the link which was provided by Philipp Heinisch";
        $conclusion1 = "Provide your annotator-ID!";
        $conclusion2 = "Without an annotator-ID no annotation!";
        $conclusion1_id = "n/a";
        $conclusion2_id =  "n/a";
        $general_frame = "nothing";
        $specific_frame = "nothing";
    } else {
        $conn_db = new mysqli("localhost", "philipp", $db_password, "philipp_umfrage");
        $stat = $conn_db->error . '<br>'. $conn_db->stat() .'<br>';

        $annotator_ID = intval($conn_db->real_escape_string($_GET["annotator_ID"]));
        
        $samples_done = $conn_db->query(
            "SELECT COUNT(*) FROM 
            (CrowdSourceArgument NATURAL JOIN CrowdSourceConclusion AS C1) JOIN CrowdSourceConclusion AS C2 ON C1.argument_ID = C2.argument_ID and C1.round = C2.round and C1.conclusion_identifier < C2.conclusion_identifier
            WHERE EXISTS(SELECT * FROM CrowdSourceAnswer WHERE CrowdSourceAnswer.argument_ID = CrowdSourceArgument.argument_ID and ((conclusion_identifier_1 = C1.conclusion_identifier and conclusion_identifier_2 = C2.conclusion_identifier) or (conclusion_identifier_1 = C2.conclusion_identifier and conclusion_identifier_2 = C1.conclusion_identifier))  and annotator_ID = ". $annotator_ID .") and C1.round IN ". $annotation_round .";"
            )->fetch_array(MYSQLI_NUM);
        $samples_done = $samples_done === FALSE ? 0 : (is_null($samples_done) ? 1 : $samples_done[0]);
        $samples_total = $conn_db->query(
            "SELECT COUNT(*) FROM 
            (CrowdSourceArgument NATURAL JOIN CrowdSourceConclusion AS C1) JOIN CrowdSourceConclusion AS C2 ON C1.argument_ID = C2.argument_ID and C1.round = C2.round and C1.conclusion_identifier <> C2.conclusion_identifier
            WHERE C1.conclusion_identifier < C2.conclusion_identifier and C1.round IN ". $annotation_round .";"
            )->fetch_array(MYSQLI_NUM);
        $samples_total =  ($samples_total === FALSE or is_null($samples_total)) ? 1 :  max(1, $samples_total[0]);

	    $result = $conn_db -> query(
            "SELECT CrowdSourceArgument.argument_ID, topic, premise,  C1.conclusion_identifier AS conclusion1_id, C1.conclusion_text AS conclusion1, C2.conclusion_identifier AS conclusion2_id, C2.conclusion_text AS conclusion2, issue_specific_frame, generic_mapped_frame, generic_inferred_frame FROM 
            (CrowdSourceArgument NATURAL JOIN CrowdSourceConclusion AS C1) JOIN CrowdSourceConclusion AS C2 ON C1.argument_ID = C2.argument_ID and C1.round = C2.round and C1.conclusion_identifier <> C2.conclusion_identifier
            WHERE NOT EXISTS(SELECT * FROM CrowdSourceAnswer WHERE CrowdSourceAnswer.argument_ID = CrowdSourceArgument.argument_ID and ((conclusion_identifier_1 = C1.conclusion_identifier and conclusion_identifier_2 = C2.conclusion_identifier) or (conclusion_identifier_1 = C2.conclusion_identifier and conclusion_identifier_2 = C1.conclusion_identifier))  and annotator_ID = ". $annotator_ID .") and C1.round IN ". $annotation_round ." 
            ORDER BY CrowdSourceArgument.argument_ID,C1.order_number, C2.order_number LIMIT 1;"
        );
        
        if ($result === FALSE or $result->num_rows == 0) {
            $sample_ID = "-1";
            $topic = "WARNING";
            $premise = "SQL-Query failed";
            $conclusion1 = "Either there is a problem with the database";
            $conclusion2 = "... or you finished all the samples properly :)";
            $conclusion1_id = "-1";
            $conclusion2_id =  "-1";
            $general_frame = "nothing";
            $specific_frame = "nothing";
        } else {
            $sample_ID = "-1";
            $topic = "...";
            $premise = "loading...";
            $conclusion1 = "loading...";
            $conclusion2 = "loading...";
            $conclusion1_id = "-1";
            $conclusion2_id =  "-1";
            $general_frame = "loading...";
            $specific_frame = "loading...";
            while ($row = $result->fetch_assoc()) {
                $sample_ID = (array_key_exists("argument_ID", $row) or !is_null($row["argument_ID"])) ? $row["argument_ID"]: 0;
                $topic = $row["topic"];
                $premise = $row["premise"];
                $conclusion1 = $row["conclusion1"];
                $conclusion2 =  $row["conclusion2"];
                $conclusion1_id = $row["conclusion1_id"];
                $conclusion2_id =  $row["conclusion2_id"];
                $generic_frame = $row["generic_mapped_frame"];
                $specific_frame = (is_null($row["issue_specific_frame"]) ? $row["generic_inferred_frame"]  : $row["issue_specific_frame"]);

                $result_prefilled_1 = $conn_db->query("SELECT a_validity, a_novelty, a_issue_specific_frame, a_generic_mapped_frame
                                                    FROM (CrowdSourceConclusion AS refConc JOIN CrowdSourceConclusion AS orgConc ON orgConc.conclusion_text = refConc.conclusion_text) JOIN CrowdSourceAnswer ON CrowdSourceAnswer.argument_ID = refConc.argument_ID and CrowdSourceAnswer.conclusion_identifier_1 = refConc.conclusion_identifier 
                                                    WHERE CrowdSourceAnswer.conclusion_identifier_2 IS NULL and orgConc.argument_ID = '". $sample_ID ."' and orgConc.conclusion_identifier = '". $conclusion1_id ."' and annotator_ID = ". $annotator_ID .";");

                if ($result_prefilled_1 !== FALSE and $result_prefilled_1->num_rows >= 1) {
                    $row = $result_prefilled_1->fetch_assoc();
                    $a_c1_validity = $row["a_validity"];
                    $a_c1_novelty = $row["a_novelty"];
                    $a_c1_issue_specific_frame = $row["a_issue_specific_frame"];
                    $a_c1_generic_mapped_frame = $row["a_generic_mapped_frame"];
                }

                $result_prefilled_2 = $conn_db->query("SELECT a_validity, a_novelty, a_issue_specific_frame, a_generic_mapped_frame
                                                    FROM (CrowdSourceConclusion AS refConc JOIN CrowdSourceConclusion AS orgConc ON orgConc.conclusion_text = refConc.conclusion_text) JOIN CrowdSourceAnswer ON CrowdSourceAnswer.argument_ID = refConc.argument_ID and CrowdSourceAnswer.conclusion_identifier_1 = refConc.conclusion_identifier 
                                                    WHERE CrowdSourceAnswer.conclusion_identifier_2 IS NULL and orgConc.argument_ID = '". $sample_ID ."' and orgConc.conclusion_identifier = '". $conclusion2_id ."' and annotator_ID = ". $annotator_ID .";");

                if ($result_prefilled_2 !== FALSE and $result_prefilled_2->num_rows >= 1) {
                    $row = $result_prefilled_2->fetch_assoc();
                    $a_c2_validity = $row["a_validity"];
                    $a_c2_novelty = $row["a_novelty"];
                    $a_c2_issue_specific_frame = $row["a_issue_specific_frame"];
                    $a_c2_generic_mapped_frame = $row["a_generic_mapped_frame"];
                }
            }
        }
        
        $conn_db->close();
    }

    switch ($generic_frame) {
        case "Economic":
            $generic_frame_description = "cost, benefits, or other financial implications";
            break;
        case "Capacity and resources":
            $generic_frame_description = "availability of physical, human or financial resources, and capacity of current systems";
            break;
        case "Morality":
            $generic_frame_description = "religious or ethical implications";
            break;
        case "Fairness and equality":
            $generic_frame_description = "balance or distribution of rights, responsibilities, and resources";
            break;
        case "Legality, constitutionality and jurisprudence":
            $generic_frame_description = "rights, freedoms, and authority of individuals, corporations, and government";
            break;
        case "Policy prescription and evaluation":
            $generic_frame_description = "discussion of specific policies aimed at addressing problems";
            break;
        case "Crime and punishment":
            $generic_frame_description = "effectiveness and implications of laws and their enforcement";
            break;
        case "Security and defence":
            $generic_frame_description = "threats to welfare of the individual, community, or nation";
            break;
        case "Health and safety":
            $generic_frame_description = "health care, sanitation, public safety";
            break;
        case "Quality of life":
            $generic_frame_description = "threats and opportunities for the individuals heath, happiness, and well-being";
            break;
        case "Cultural identity":
            $generic_frame_description = "traditions, customs, or values of a social group in relation to a policy issues";
            break;
        case "Public opinion":
            $generic_frame_description = "attitudes and opinions of the general public, including polling and demographics";
            break;
        case "Political":
            $generic_frame_description = "considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters";
            break;
        case "External regulation and reputation":
            $generic_frame_description = "internal reputation or foreign policy of (e.g.) the USA";
            break;
        default:
            $generic_frame_description = "No further desciption available";
    }

    $topic = is_null($error_msg) ? $topic : $error_msg;
?>

<!DOCTYPE html>
<head>
    <title>You're in annotation mood (<?php echo $_GET["annotator_ID"] ?>)</title>
    <script>
            function Validity(l, v="compare") {
                if (v == "compare") {
                    if (l == 'left') {
                        document.getElementById("left").style.backgroundColor = "#90EE90";
                        document.getElementById("right").style.backgroundColor = "yellow";
                    } else if (l == 'right') {
                        document.getElementById("right").style.backgroundColor = "#90EE90";
                        document.getElementById("left").style.backgroundColor = "yellow";
                    } else {
                        document.getElementById("left").style.backgroundColor = "yellow";
                        document.getElementById("right").style.backgroundColor = "yellow";
                    }
                } else if (v == 1) {
                    document.getElementById(l).style.color = "DarkGreen";
                } else if (v == 0) {
                    document.getElementById(l).style.color = "black";
                } else {
                    document.getElementById(l).style.color = "DarkRed";
                }

                if (document.getElementById("c1_validity_1").checked && document.getElementById("c2_validity_-1").checked) {
                    document.getElementById("c2c_validity_-1").checked = true;
                    Validity("left", "compare")
                } else if (document.getElementById("c2_validity_1").checked && document.getElementById("c1_validity_-1").checked) {
                    document.getElementById("c2c_validity_1").checked = true;
                    Validity("right", "compare")
                }
            }

            function  Novelty(l, v="compare") {
                if (v == "compare") {
                    if (l == 'left') {
                        document.getElementById("left").style.fontWeight = "bold";
                        document.getElementById("right").style.fontWeight = "normal";
                    } else if(l == 'right') {
                        document.getElementById("right").style.fontWeight = "bold";
                        document.getElementById("left").style.fontWeight = "normal";
                    } else {
                        document.getElementById("left").style.fontWeight = "normal";
                        document.getElementById("right").style.fontWeight = "normal";
                    }
                } else if (v == 1) {
                    document.getElementById(l).style.textDecoration = "underline gray";
                } else if (v == 0) {
                    document.getElementById(l).style.textDecoration = "initial";
                } else {
                    document.getElementById(l).style.textDecoration = "line-through";
                }

                if (document.getElementById("c1_novelty_1").checked && document.getElementById("c2_novelty_-1").checked) {
                    document.getElementById("c2c_novelty_-1").checked = true;
                    Novelty("left", "compare")
                } else if (document.getElementById("c2_novelty_1").checked && document.getElementById("c1_novelty_-1").checked) {
                    document.getElementById("c2c_novelty_1").checked = true;
                    Novelty("right", "compare")
                }
            }

            function  FrameSpec(l, v="compare") {
                if (v == "compare") {
                    if (l == 'left') {
                        document.getElementById("left").style.borderWidth = "3px";
                        document.getElementById("right").style.borderWidth = "1px";
                    } else if(l == 'right') {
                        document.getElementById("right").style.borderWidth = "3px";
                        document.getElementById("left").style.borderWidth = "1px";
                    } else {
                        document.getElementById("left").style.borderWidth = "1px";
                        document.getElementById("right").style.borderWidth = "1px";
                    }
                } else if (v == 1) {
                    document.getElementById(l).style.borderBottomColor = "DarkGreen";
                } else if (v == 0) {
                    document.getElementById(l).style.borderBottomColor = "gray";
                } else {
                    document.getElementById(l).style.borderBottomColor = "DarkRed";
                }

                if (document.getElementById("c1_specificFraming_1").checked && document.getElementById("c2_specificFraming_-1").checked) {
                    document.getElementById("c2c_specificFraming_-1").checked = true;
                    FrameSpec("left", "compare")
                } else if (document.getElementById("c2_specificFraming_1").checked && document.getElementById("c1_specificFraming_-1").checked) {
                    document.getElementById("c2c_specificFraming_1").checked = true;
                    FrameSpec("right", "compare")
                }
            }

            function  FrameGen(l, v="compare") {
                if (v == "compare") {
                    if (l == 'left') {
                        document.getElementById("left").style.borderStyle = "solid";
                        document.getElementById("right").style.borderStyle = "dashed";
                    } else if(l == 'right') {
                        document.getElementById("right").style.borderStyle = "solid";
                        document.getElementById("left").style.borderStyle = "dashed";
                    } else {
                        document.getElementById("left").style.borderStyle = "dashed";
                        document.getElementById("right").style.borderStyle = "dashed";
                    }
                } else if (v == 1) {
                    document.getElementById(l).style.borderTopColor = "DarkGreen";
                } else if (v == 0) {
                    document.getElementById(l).style.borderTopColor = "gray";
                } else {
                    document.getElementById(l).style.borderTopColor = "DarkRed";
                }

                if (document.getElementById("c1_generalFraming_1").checked && document.getElementById("c2_generalFraming_-1").checked) {
                    document.getElementById("c2c_generalFraming_-1").checked = true;
                    FrameGen("left", "compare")
                } else if (document.getElementById("c2_generalFraming_1").checked && document.getElementById("c1_generalFraming_-1").checked) {
                    document.getElementById("c2c_generalFraming_1").checked = true;
                    FrameGen("right", "compare")
                }
            }
    </script>
    <style>
        .column_2 {
            float: left;
            width: 45%;
        }
        @media screen and (max-width: 400) {
            .column_2 {
                width: 100%;
            }
        }
        .column_3 {
            float: left;
            width: 33.33%;
        }
        @media screen and (max-width: 600) {
            .column_3 {
                width: 100%;
            }
        }

        /* Clear floats after the columns */
        .row:after {
            content: "";
            display: table;
            clear: both;
        }
    </style>
</head>
<body style="text-align: center; max-width: 1000px; margin: auto;">
    <h1>Sample <?php echo $sample_ID; ?></h1>
    <progress id="progress_annotation" value="<?php echo round($samples_done*1000/$samples_total); ?>" max="1000"> <?php echo $samples_done ." out of ". $samples_total; ?> </progress>
    <h2><?php echo $topic; ?></h2>
    <div style="width: 98%; background-color: lightgray; border: 1px solid black; border-radius: 20px; padding: 10px; margin-top: 12px; margin-bottom: 20px; display: inline-block;"><b>Premise:</b> <?php echo $premise; ?></div>
    <div class="row">
        <div id="left" class="column_2" style="background-color: yellow; border: 1px dashed black; border-radius: 20px; padding: 20px; margin-right: 5px;"><b>Conclusion 1:</b> <?php echo $conclusion1; ?></div>
        <div id="right" class="column_2" style="background-color: yellow; border: 1px dashed black; border-radius: 20px; padding: 20px; margin-left: 5px;"><b>Conclusion 2:</b> <?php echo $conclusion2; ?></div>
    </div>
    <h2>Let's rate ;)</h2>
    <form action="<?php echo $_SERVER["REQUEST_URI"]; ?>" method="POST" autocomplete="off">
        <h3>Validity: Conclusion is justified based on the premise</h3>
        <div class="row">
            <div class="column_3">
                <h4>Conclusion 1</h4>
                <ol>
                    <li><input type="radio" value="1" name="c1_validity" id="c1_validity_1" required onclick="Validity('left', 1);" title="I agree" <?php if($a_c1_validity === "1") { ?>checked<?php }?>>yes</li>
                    <li><input type="radio" value="0" name="c1_validity" id="c1_validity_0" required onclick="Validity('left', 0);" title="0" <?php if($a_c1_validity === "0") { ?>checked<?php }?>>I can't decide</li>
                    <li><input type="radio" value="-1" name="c1_validity" id="c1_validity_-1" required onclick="Validity('left', -1);" title="I disagree" <?php if($a_c1_validity === "-1") { ?>checked<?php }?>>no</li>
                </ol>
            </div>
            <div class="column_3">
                <h4>Conclusion 1 vs. Conclusion 2</h4>
                <ol>
                    <li><input type="radio" value="-1" name="c2c_validity" id="c2c_validity_-1" required onclick="Validity('left');" title="Conclusion 1" <?php if($a_c1_validity === "1" && $a_c2_validity === "-1") { ?>checked<?php }?>>Conclusion 1 is more valid</li>
                    <li><input type="radio" value="0" name="c2c_validity" id="c2c_validity_-0" required onclick="Validity('none');" title="NONE" <?php if($conclusion1 == $conclusion2) { ?>checked<?php }?>>Both are equally bad/ good</li>
                    <li><input type="radio" value="1" name="c2c_validity" id="c2c_validity_1" required onclick="Validity('right');" title="Conclusion 2" <?php if($a_c1_validity === "-1" && $a_c2_validity === "1") { ?>checked<?php }?>>Conclusion 2 is more valid</li>
                </ol>
            </div>
            <div class="column_3">
                <h4>Conclusion 2</h4>
                <ol>
                    <li><input type="radio" value="1" name="c2_validity" id="c2_validity_1" required onclick="Validity('right', 1);" title="I agree" <?php if($a_c2_validity === "1") { ?>checked<?php }?>>yes</li>
                    <li><input type="radio" value="0" name="c2_validity" id="c2_validity_0" required onclick="Validity('right', 0);" title="0" <?php if($a_c2_validity === "0") { ?>checked<?php }?>>I can't decide</li>
                    <li><input type="radio" value="-1" name="c2_validity" id="c2_validity_-1" required onclick="Validity('right', -1);" title="I disagree" <?php if($a_c2_validity === "-1") { ?>checked<?php }?>>no</li>
                </ol>
            </div>
        </div>
        <h3>Novelty: Conclusion introduces premise-related novel content (is, e.g., not a paraphrased repetition of (a part of) the premise)</h3>
        <div class="row">
            <div class="column_3">
                <h4>Conclusion 1</h4>
                <ol>
                    <li><input type="radio" value="1" name="c1_novelty" id="c1_novelty_1" required onclick="Novelty('left', 1);" title="I agree" <?php if($a_c1_novelty === "1") { ?>checked<?php }?>>yes</li>
                    <li><input type="radio" value="0" name="c1_novelty" id="c1_novelty_0" required onclick="Novelty('left', 0);" title="0" <?php if($a_c1_novelty === "0") { ?>checked<?php }?>>I can't decide</li>
                    <li><input type="radio" value="-1" name="c1_novelty" id="c1_novelty_-1" required onclick="Novelty('left', -1);" title="I disagree" <?php if($a_c1_novelty === "-1") { ?>checked<?php }?>>no</li>
                </ol>
            </div>
            <div class="column_3">
                <h4>Conclusion 1 vs. Conclusion 2</h4>
                <ol>
                    <li><input type="radio" value="-1" name="c2c_novelty" id="c2c_novelty_-1" required onclick="Novelty('left');" title="Conclusion 1" <?php if($a_c1_novelty === "1" && $a_c2_novelty === "-1") { ?>checked<?php }?>>Conclusion 1 contains more novel (proper) content</li>
                    <li><input type="radio" value="0" name="c2c_novelty" id="c2c_novelty_0" required onclick="Novelty('none');" title="NONE" <?php if($conclusion1 == $conclusion2) { ?>checked<?php }?>>Both contain the equal amount</li>
                    <li><input type="radio" value="1" name="c2c_novelty" id="c2c_novelty_1" required onclick="Novelty('right');" title="Conclusion 2" <?php if($a_c1_novelty === "-1" && $a_c2_novelty === "1") { ?>checked<?php }?>>Conclusion 2 contains more novel (proper) content</li>
                </ol>
            </div>
            <div class="column_3">
                <h4>Conclusion 2</h4>
                <ol>
                    <li><input type="radio" value="1" name="c2_novelty" id="c2_novelty_1" required onclick="Novelty('right', 1);" title="I agree" <?php if($a_c2_novelty === "1") { ?>checked<?php }?>>yes</li>
                    <li><input type="radio" value="0" name="c2_novelty" id="c2_novelty_0" required onclick="Novelty('right', 0);" title="0" <?php if($a_c2_novelty === "0") { ?>checked<?php }?>>I can't decide</li>
                    <li><input type="radio" value="-1" name="c2_novelty" id="c2_novelty_-1" required onclick="Novelty('right', -1);" title="I disagree" <?php if($a_c2_novelty === "-1") { ?>checked<?php }?>>no</li>
                </ol>
            </div>
        </div>
        <?php if(!is_null($generic_frame) and $generic_frame !== "NULL") { ?>
        <h3> Generic perspective &raquo;<?php echo $generic_frame; ?>&laquo;</h3>
        The conclusion is directed towards the perspective <abbr title="<?php echo $generic_frame_description; ?>">&raquo;<?php echo $generic_frame; ?>&laquo;</abbr>.
        <div class="row">
            <div class="column_3">
                <h4>Conclusion 1</h4>
                <ol>
                    <li><input type="radio" value="1" name="c1_generalFraming" id="c1_generalFraming_1" required onclick="FrameGen('left', 1);" title="I agree" <?php if($a_c1_generic_mapped_frame === "1") { ?>checked<?php }?>>yes</li>
                    <li><input type="radio" value="0" name="c1_generalFraming" id="c1_generalFraming_0" required onclick="FrameGen('left', 0);" title="0" <?php if($a_c1_generic_mapped_frame === "0") { ?>checked<?php }?>>I can't decide</li>
                    <li><input type="radio" value="-1" name="c1_generalFraming" id="c1_generalFraming_-1" required onclick="FrameGen('left', -1);" title="I disagree" <?php if($a_c1_generic_mapped_frame === "-1") { ?>checked<?php }?>>no</li>
                </ol>
            </div>
            <div class="column_3">
                <h4>Conclusion 1 vs. Conclusion 2</h4>
                <ol>
                    <li><input type="radio" value="-1" name="c2c_generalFraming" id="c2c_generalFraming_-1" required onclick="FrameGen('left');" title="Conclusion 1" <?php if($a_c1_generic_mapped_frame === "1" && $a_c2_generic_mapped_frame === "-1") { ?>checked<?php }?>>Conclusion 1 fits better</li>
                    <li><input type="radio" value="0" name="c2c_generalFraming" id="c2c_generalFraming_0" required onclick="FrameGen('none');" title="NONE" <?php if($conclusion1 == $conclusion2) { ?>checked<?php }?>>Both fit equally bad/ good</li>
                    <li><input type="radio" value="1" name="c2c_generalFraming" id="c2c_generalFraming_1" required onclick="FrameGen('right');" title="Conclusion 2" <?php if($a_c1_generic_mapped_frame === "-1" && $a_c2_generic_mapped_frame === "1") { ?>checked<?php }?>>Conclusion 2 fits better</li>
                </ol>
            </div>
            <div class="column_3">
                <h4>Conclusion 2</h4>
                <ol>
                    <li><input type="radio" value="1" name="c2_generalFraming" id="c2_generalFraming_1" required onclick="FrameGen('right', 1);" title="I agree" <?php if($a_c2_generic_mapped_frame === "1") { ?>checked<?php }?>>yes</li>
                    <li><input type="radio" value="0" name="c2_generalFraming" id="c2_generalFraming_0" required onclick="FrameGen('right', 0);" title="0" <?php if($a_c2_generic_mapped_frame === "0") { ?>checked<?php }?>>I can't decide</li>
                    <li><input type="radio" value="-1" name="c2_generalFraming" id="c2_generalFraming_-1" required onclick="FrameGen('right', -1);" title="I disagree" <?php if($a_c2_generic_mapped_frame === "-1") { ?>checked<?php }?>>no</li>
                </ol>
            </div>
        </div>
        <?php }
        if(!is_null($specific_frame) and $specific_frame !== "NULL") { ?>
        <h3>Specific perspective &raquo;<?php echo $specific_frame; ?>&laquo;</h3>
        The conclusion is directed towards the perspective &raquo;<?php echo $specific_frame; ?>&laquo;.
        <div class="row">
            <div class="column_3">
                <h4>Conclusion 1</h4>
                <ol>
                    <li><input type="radio" value="1" name="c1_specificFraming" id="c1_specificFraming_1" required onclick="FrameSpec('left', 1);" title="I agree" <?php if($a_c1_issue_specific_frame === "1") { ?>checked<?php }?>>yes</li>
                    <li><input type="radio" value="0" name="c1_specificFraming" id="c1_specificFraming_0" required onclick="FrameSpec('left', 0);" title="0" <?php if($a_c1_issue_specific_frame === "0") { ?>checked<?php }?>>I can't decide</li>
                    <li><input type="radio" value="-1" name="c1_specificFraming" id="c1_specificFraming_-1" required onclick="FrameSpec('left', -1);" title="I disagree" <?php if($a_c1_issue_specific_frame === "-1") { ?>checked<?php }?>>no</li>
                </ol>
            </div>
            <div class="column_3">
                <h4>Conclusion 1 vs. Conclusion 2</h4>
                <ol>
                    <li><input type="radio" value="-1" name="c2c_specificFraming" id="c2c_specificFraming_-1" required onclick="FrameSpec('left');" title="Conclusion 1" <?php if($a_c1_issue_specific_frame === "1" && $a_c2_issue_specific_frame === "-1") { ?>checked<?php }?>>Conclusion 1 fits better</li>
                    <li><input type="radio" value="0" name="c2c_specificFraming" id="c2c_specificFraming_0" required onclick="FrameSpec('none');" title="NONE" <?php if($conclusion1 == $conclusion2) { ?>checked<?php }?>>Both fit equally bad/ good</li>
                    <li><input type="radio" value="1" name="c2c_specificFraming" id="c2c_specificFraming_1" required onclick="FrameSpec('right');" title="Conclusion 2" <?php if($a_c1_issue_specific_frame === "-1" && $a_c2_issue_specific_frame === "1") { ?>checked<?php }?>>Conclusion 2 fits better</li>
                </ol>
            </div>
            <div class="column_3">
                <h4>Conclusion 2</h4>
                <ol>
                    <li><input type="radio" value="1" name="c2_specificFraming" id="c2_specificFraming_1" required onclick="FrameSpec('right', 1);" title="I agree" <?php if($a_c2_issue_specific_frame === "1") { ?>checked<?php }?>>yes</li>
                    <li><input type="radio" value="0" name="c2_specificFraming" id="c2_specificFraming_0" required onclick="FrameSpec('right', 0);" title="0" <?php if($a_c2_issue_specific_frame === "0") { ?>checked<?php }?>>I can't decide</li>
                    <li><input type="radio" value="-1" name="c2_specificFraming" id="c2_specificFraming_-1" required onclick="FrameSpec('right', -1);" title="I disagree" <?php if($a_c2_issue_specific_frame === "-1") { ?>checked<?php }?>>no</li>
                </ol>
            </div>
        </div>
        <?php } ?>
        <textarea cols="80" rows="2" name="comments" maxlength="256" placeholder="Any questions/ comments to this sample? (optional)"></textarea>
        <br>
        <input type="hidden" value="<?php echo $sample_ID; ?>" name="sample_ID">
        <input type="hidden" value="<?php echo $conclusion1_id; ?>" name="conclusion1_id">
        <input type="hidden" value="<?php echo $conclusion2_id; ?>" name="conclusion2_id">
        <input type="hidden" value="<?php echo time(); ?>" name="timeStart">
        <br>
        <input type="submit" value=">>> Save & next >>>">
    </form>
    <hr>
    <div style="display: inline-block; width: 100%;">
        <h2>Instructions</h2>
        <p>We ask you to rate conclusions (in a pair-wise manner) in a discussion. To this end, you have for each argument the topic in the title, a premise (the explain- / give-reasons-part of an argument) and two conclusions (hopefully) matching the premise. However - which one is (more) appropriate? Which conclusion fits the premise better?</p>

        <p>Premise --&gt; conclusion 1 ? AND/OR Premise --&gt; conclusion 2?</p>

        <h4>To have differentiated voting, you have to decide on 3+1 aspects.</h4>

        <ol>
            <li><b>Validity:</b> Which conclusion is (more) reasonable / (more) appropriate? So, what is (more) likely? Conclusion 1 follows given the premise and/or conclusion 2? <i>Hint: a repetition of the premise would be totally valid, it's a so-called tautology.</i></li>
            <li><b>Novelty:</b> Which conclusion contains more novel information? <i>(the novel information must be premise-related - a random sentence of another topic probably doesn't contain novel content with respect to the premise.)</i> Just copying (or rephrasing) parts of the premise is too easy (should get no vote rather). Here we seek the conclusion with the most novel content, with the most conclusion-information-gain.</li>
            <li><b>Perspective &raquo;xxx&laquo;:</b> You can argue from different perspectives. For example, if you discuss raising taxes, you can emphasize the side of the tax-payers or the government's side or think about moral viewpoints. All these perspectives are called &raquo;Frames&laquo;. Here, you should rate how much the desired perspective occurs in the <i>conclusions</i> (without considering the appropriateness to the premise again). A conclusion that fits more (obviously) into the desired perspective should gain the vote.</li>
        </ol>
        <h3>Positive and negative examples</h3>
        <h4>Positive (you should do it in such a way)</h4>
        <img src="./img/positive_example-pairwise.jpg" width="50%">
        <ul>
            <li>differentiated voting: although conclusion 2 is the more trivial one (somewhat a rephrased interpretation of the premise), conclusion 1 is more novel and a real inference, hence it receives 2/4 votes</li>
            <li>you read each text carefully, thinking about it and hence, you noticed conclusion 2 is a little bit clearer about research and conclusion 1 captures more a big question of life</li>
        </ul>
        <h4>Negative (no, no, no...)</h4>
        <img src="./img/negative_example-pairwise.jpg" width="50%">
        <ul>
            <li>lazy voting (here only for conclusion 2 or tie) - yes, sometimes one conclusion outperforms the other in <b>all</b> categories, but this is rare</li>
            <li>maybe you're thinking: "I'm a atheist, therefore nothing with God!" and therefore, you voted for conclusion 2 - however, we don't ask for your personal opinion about the topic, only stick to the premise and the conclusion - and the premise is a perfect base for conclusion 1.</li>
        </ul>
    </div>
</body>