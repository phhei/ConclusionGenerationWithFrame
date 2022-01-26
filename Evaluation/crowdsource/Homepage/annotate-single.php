<?php
    require "password.php";

    $annotation_round = 1;

    if($_POST and array_key_exists("annotator_ID", $_GET)) {
        $conn_db = new mysqli("localhost", "philipp", $db_password, "philipp_umfrage");
        $stat = $conn_db->error . '<br>'. $conn_db->stat() .'<br>';

        $annotator_ID = $conn_db->real_escape_string($_GET["annotator_ID"]);
        $sample_ID_commit = $conn_db->real_escape_string($_POST["sample_ID"]);
        $sample_conclusion_ID_commit = $conn_db->real_escape_string($_POST["sample_conclusion_ID"]);

        if ($conn_db->query("SELECT COUNT(*) FROM CrowdSourceAnswer WHERE annotator_ID = ". $annotator_ID ." and argument_ID = '". $sample_ID_commit . "' and conclusion_identifier_1 = '". $sample_conclusion_ID_commit ."' and conclusion_identifier_2 IS NULL;")->fetch_array(MYSQLI_NUM)[0] >= 1) {
            $error_msg = "You already annotated this sample!";
        } else {
            $validity = $conn_db->real_escape_string($_POST["validity"]);
            $novelty = $conn_db->real_escape_string($_POST["novelty"]);
            $generalFraming = array_key_exists("generalFraming", $_POST) ? $conn_db->real_escape_string($_POST["generalFraming"]) : "NULL";
            $specificFraming = array_key_exists("specificFraming", $_POST) ? $conn_db->real_escape_string($_POST["specificFraming"]) : "NULL";
            $time = time() - $_POST["timeStart"];
            $comments = $conn_db->real_escape_string($_POST["comments"]);
            $comments = (is_null($comments) or $comments == "") ? "no comment" : $comments;

            $query = "INSERT INTO CrowdSourceAnswer (annotator_ID, argument_ID, conclusion_identifier_1, timeInS, a_validity, a_novelty, a_issue_specific_frame, a_generic_mapped_frame, a_comment) VALUES
                                            (".$annotator_ID.", \"".$sample_ID_commit."\", \"".$sample_conclusion_ID_commit."\", ". $time .", ".$validity .", ".$novelty.", ".$specificFraming.", ".$generalFraming.", \"".$comments."\");";

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
        $sample_conclusion_ID = "n/a";
        $samples_done = 0;
        $samples_total = 1;
        $topic = "No annotator-ID provided!";
        $premise = "Please use the link which was provided by Philipp Heinisch";
        $conclusion = "Provide your annotator-ID! Without an annotator-ID no annotation!";
        $generic_frame = null;
        $specific_frame = null;
    } else {
        $conn_db = new mysqli("localhost", "philipp", $db_password, "philipp_umfrage");
        $stat = $conn_db->error . '<br>'. $conn_db->stat() .'<br>';

        $annotator_ID = intval($conn_db->real_escape_string($_GET["annotator_ID"]));
        
        $samples_done = $conn_db->query("SELECT COUNT(*) FROM CrowdSourceConclusion WHERE round=". $annotation_round ." and EXISTS
                                        (SELECT * FROM CrowdSourceAnswer WHERE CrowdSourceConclusion.argument_ID = CrowdSourceAnswer.argument_ID and CrowdSourceConclusion.conclusion_identifier = CrowdSourceAnswer.conclusion_identifier_1 and CrowdSourceAnswer.conclusion_identifier_2 IS NULL and annotator_ID = ". $annotator_ID .");")->fetch_array(MYSQLI_NUM);
        $samples_done = $samples_done === FALSE ? 0 : (is_null($samples_done) ? 1 : $samples_done[0]);
        $samples_total = $conn_db->query("SELECT COUNT(*) FROM CrowdSourceConclusion WHERE round=". $annotation_round)->fetch_array(MYSQLI_NUM);
        $samples_total =  ($samples_total === FALSE or is_null($samples_total)) ? 1 :  max(1, $samples_total[0]);

	    $result = $conn_db -> query("SELECT * FROM CrowdSourceArgument NATURAL JOIN CrowdSourceConclusion WHERE round=". $annotation_round ." and NOT EXISTS
                                    (SELECT * FROM CrowdSourceAnswer WHERE CrowdSourceConclusion.argument_ID = CrowdSourceAnswer.argument_ID and CrowdSourceConclusion.conclusion_identifier = CrowdSourceAnswer.conclusion_identifier_1 and CrowdSourceAnswer.conclusion_identifier_2 IS NULL and annotator_ID = ". $annotator_ID .")
                                    ORDER BY order_number LIMIT 1;");
        
        if ($result === FALSE or $result->num_rows == 0) {
            $sample_ID = "-1";
            $sample_conclusion_ID = "-1";
            $topic = "WARNING";
            $premise = "SQL-Query failed";
            $conclusion = "Either there is a problem with the database or you finished all the samples properly :)";
            $generic_frame = "nothing";
            $specific_frame = "nothing";
            $a_validity = 0;
            $a_novelty = 0;
            $a_issue_specific_frame = 0;
            $a_generic_mapped_frame = 0;
        } else {
            $sample_ID = "-1";
            $sample_conclusion_ID = "-1";
            $topic = "...";
            $premise = "loading...";
            $conclusion = "loading...";
            $generic_frame = "loading...";
            $specific_frame = "loading...";
            while ($row = $result->fetch_assoc()) {
                $sample_ID = (array_key_exists("argument_ID", $row) or !is_null($row["argument_ID"])) ? $row["argument_ID"]: 0;
                $sample_conclusion_ID = (array_key_exists("conclusion_identifier", $row) or !is_null($row["conclusion_identifier"])) ? $row["conclusion_identifier"]: 0;
                $topic = $row["topic"];
                $premise = $row["premise"];
                $conclusion = $row["conclusion_text"];
                $generic_frame = $row["generic_mapped_frame"];
                $specific_frame = (is_null($row["issue_specific_frame"]) ? $row["generic_inferred_frame"]  : $row["issue_specific_frame"]);

                $result_prefilled = $conn_db->query("SELECT a_validity, a_novelty, a_issue_specific_frame, a_generic_mapped_frame 
                                                    FROM (CrowdSourceConclusion AS refConc JOIN CrowdSourceConclusion AS orgConc ON orgConc.conclusion_text = refConc.conclusion_text) JOIN CrowdSourceAnswer ON CrowdSourceAnswer.argument_ID = refConc.argument_ID and CrowdSourceAnswer.conclusion_identifier_1 = refConc.conclusion_identifier 
                                                    WHERE CrowdSourceAnswer.conclusion_identifier_2 IS NULL and orgConc.argument_ID = '". $sample_ID ."' and orgConc.conclusion_identifier = '". $sample_conclusion_ID ."' and annotator_ID = ". $annotator_ID .";");

                if ($result_prefilled !== FALSE and $result_prefilled->num_rows >= 1) {
                    $row = $result_prefilled->fetch_assoc();
                    $a_validity = $row["a_validity"];
                    $a_novelty = $row["a_novelty"];
                    $a_issue_specific_frame = $row["a_issue_specific_frame"];
                    $a_generic_mapped_frame = $row["a_generic_mapped_frame"];
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
            function Validity(l) {
                if (l == 1) {
                    document.getElementById('conclusion').style.backgroundColor = "#90EE90";
                } else if (l == 0) {
                    document.getElementById('conclusion').style.backgroundColor = "white";
                } else {
                    document.getElementById('conclusion').style.backgroundColor = "yellow";
                }
            }

            function  Novelty(l) {
                if (l == 1) {
                    document.getElementById('conclusion').style.fontWeight = "bold";
                } else if(l == 0) {
                    document.getElementById('conclusion').style.fontWeight = "normal";
                } else {
                    document.getElementById('conclusion').style.fontWeight = 100;
                }
            }

            function  FrameSpec(l) {
                if (l == 1) {
                    document.getElementById('conclusion').style.borderWidth = "3px";
                } else if(l == 0) {
                    document.getElementById('conclusion').style.borderWidth = "2px";
                } else {
                    document.getElementById('conclusion').style.borderWidth = "1px";
                }
            }

            function  FrameGen(l) {
                if (l == 1) {
                    document.getElementById('conclusion').style.borderStyle = "solid";
                } else if(l == 0) {
                    document.getElementById('conclusion').style.borderStyle = "dashed";
                } else {
                    document.getElementById('conclusion').style.borderStyle = "dotted";
                }
            }
    </script>
</head>
<body style="text-align: center; max-width: 1000px; margin: auto;">
    <h1>Sample <?php echo $sample_ID; ?></h1>
    <progress id="progress_annotation" value="<?php echo $samples_done*1000/$samples_total ?>" max="1000"> <?php echo round($samples_done*100/$samples_total) ?>% </progress>
    <h2><?php echo $topic; ?></h2>
    <div style="width: 98%; background-color: lightgray; border: 1px solid black; border-radius: 20px; padding: 10px; margin-top: 12px; margin-bottom: 20px; display: inline-block;"><b>Premise:</b> <?php echo $premise; ?></div>
    <div id="conclusion" style="width: 98%; background-color: yellow; border: 1px dashed black; border-radius: 20px; padding: 10px;"><b>Conclusion:</b> <?php echo $conclusion; ?></div>
    <form action="annotate-single.php?annotator_ID=<?php echo $_GET["annotator_ID"]; ?>" method="POST" autocomplete="off">
        <h3>Validity: Conclusion is justified based on the premise</h3>
        <ol>
            <li><input type="radio" value="1" name="validity" required onclick="Validity(1);" title="I agree" <?php if($a_validity === "1") { ?>checked<?php }?>>yes</li>
            <li><input type="radio" value="0" name="validity" required onclick="Validity(0);" title="0" <?php if($a_validity === "0") { ?>checked<?php }?>>I can't decide (try to avoid this option - only if you're really unsure)</li>
            <li><input type="radio" value="-1" name="validity" required onclick="Validity(-1);" title="I disagree" <?php if($a_validity === "-1") { ?>checked<?php }?>>no</li>
        </ol>
        <h3>Novelty: Conclusion introduces premise-related novel content (is, e.g., not a paraphrased repetition of (a part of) the premise)</h3>
        <ol>
            <li><input type="radio" value="1" name="novelty" required onclick="Novelty(1);" title="I agree" <?php if($a_novelty === "1") { ?>checked<?php }?>>yes</li>
            <li><input type="radio" value="0" name="novelty" required onclick="Novelty(0);" title="0" <?php if($a_novelty === "0") { ?>checked<?php }?>>I can't decide (try to avoid this option - only if you're really unsure)</li>
            <li><input type="radio" value="-1" name="novelty" required onclick="Novelty(-1);" title="I disagree" <?php if($a_novelty === "-1") { ?>checked<?php }?>>no</li>
        </ol>
        <?php if(!is_null($generic_frame) and $generic_frame !== "NULL") { ?>
        <h3> Generic perspective &raquo;<?php echo $generic_frame; ?>&laquo;</h3>
        The conclusion is directed towards the perspective <abbr title="<?php echo $generic_frame_description; ?>">&raquo;<?php echo $generic_frame; ?>&laquo;</abbr>.
        <ol>
            <li><input type="radio" value="1" name="generalFraming" required onclick="FrameGen(1);" title="I agree" <?php if($a_generic_mapped_frame === "1") { ?>checked<?php }?>>yes</li>
            <li><input type="radio" value="0" name="generalFraming" required onclick="FrameGen(0);" title="NONE" <?php if($a_generic_mapped_frame === "0") { ?>checked<?php }?>>I can't decide (try to avoid this option - only if you're really unsure)</li>
            <li><input type="radio" value="-1" name="generalFraming" required onclick="FrameGen(-1);" title="I disagree" <?php if($a_generic_mapped_frame === "-1") { ?>checked<?php }?>>no</li>
        </ol>
        <?php }
        if(!is_null($specific_frame) and $specific_frame !== "NULL") { ?>
        <h3>Specific perspective &raquo;<?php echo $specific_frame; ?>&laquo;</h3>
        The conclusion is directed towards the perspective &raquo;<?php echo $specific_frame; ?>&laquo;.
        <ol>
            <li><input type="radio" value="1" name="specificFraming" required onclick="FrameSpec(1);" title="I agree" <?php if($a_issue_specific_frame === "1") { ?>checked<?php }?>>yes</li>
            <li><input type="radio" value="0" name="specificFraming" required onclick="FrameSpec(0);" title="NONE" <?php if($a_issue_specific_frame === "0") { ?>checked<?php }?>>I can't decide (try to avoid this option - only if you're really unsure)</li>
            <li><input type="radio" value="-1" name="specificFraming" required onclick="FrameSpec(-1);" title="I disagree" <?php if($a_issue_specific_frame === "-1") { ?>checked<?php }?>>no</li>
        </ol>
        <?php } ?>
        <textarea cols="80" rows="2" name="comments" maxlength="256" placeholder="Any questions/ comments to this sample? (optional)"></textarea>
        <br>
        <input type="hidden" value="<?php echo $sample_ID; ?>" name="sample_ID">
        <input type="hidden" value="<?php echo $sample_conclusion_ID; ?>" name="sample_conclusion_ID">
        <input type="hidden" value="<?php echo time(); ?>" name="timeStart">
        <br>
        <input type="submit" value=">>> Save & next >>>">
    </form>
    <hr>
    <div style="display: inline-block; width: 100%;">
        <h2>Instructions</h2>
        <p>We ask you to rate conclusions in a discussion. To this end, you have for each argument the topic in the title, a premise (the explain- / give-reasons-part of an argument) and a conclusions (hopefully) matching the premise. However - is the conclusion really appropriate?</p>

        <h4>To have differentiated voting, you have to decide on 3+1 aspects.</h4>

        <ol>
            <li><b>Validity:</b> Is the conclusion reasonable / appropriate? So, what is that likely or true: The conclusion follows given the premise? Hint: a repetition of the premise would be totally valid, it's a so-called tautology.</li>
            <li><b>Novelty:</b> Contains the conclusion novel (appropriate) information? <i>(the novel information must be premise-related - a random sentence of another topic probably doesn't contain novel content with respect to the premise.)</i> Just copying (or rephrasing already stated) parts of the premise should get a "no"-vote.</li>
            <li><b>Perspective &raquo;xxx&laquo;:</b> You can argue from different perspectives. For example, if you discuss raising taxes, you can emphasize the side of the tax-payers or the government's side or think about moral viewpoints. All these perspectives are called &raquo;Frames&laquo;. Here, you should rate how much the desired perspective occurs in the <i>conclusion</i> (without considering the appropriateness to the premise again). A conclusion that fits (obviously) into the desired perspective should gain the positive vote.</li>
        </ol>
        <h3>Positive and negative examples</h3>
        <h4>Positive (you should do it in such a way)</h4>
        <img src="../crowdsource/img/positive_example-single.jpg" width="50%">
        <ul>
            <li>differentiated voting: although the conclusion is appropiate based on the premise, it doesn't receive all "yes"-votes since the conclusion doesn't say anything in the direction of health (generic frame)</li>
            <li>you read each text carefully, thinking about it and hence, you noticed the inference step from the premise to the conclusion and the relatedness to &raquo;the big question of life&laquo;</li>
        </ul>
        <h4>Negative (no, no, no...)</h4>
        <img src="../crowdsource/img/negative_example-single.jpg" width="50%">
        <ul>
            <li>lazy voting (here only "no"s) - yes, sometimes one conclusion is good/ bad in <b>all</b> categories, but this is not often</li>
            <li>maybe you're thinking: "I'm a atheist, therefore nothing with God!" and therefore, you voted against the conclusion - however, we don't ask for your personal opinion about the topic, only stick to the premise and the conclusion - and the premise is a base for the conclusion.</li>
        </ul>
    </div>
</body>