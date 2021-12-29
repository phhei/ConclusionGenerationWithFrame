<?php
    require "password.php";

    if($_POST and array_key_exists("annotator_ID", $_GET)) {
        $conn_db = new mysqli("localhost", "philipp", $db_password, "philipp_umfrage");
        $stat = $conn_db->error . '<br>'. $conn_db->stat() .'<br>';

        $annotator_ID = $conn_db->real_escape_string($_GET["annotator_ID"]);
        $sample_ID_commit = $conn_db->real_escape_string($_POST["sample_ID"]);

        if ($conn_db->query("SELECT COUNT(*) FROM CrowdSourceAnswer WHERE annotator_ID = ". $annotator_ID ." and test_ID = ". $sample_ID_commit . ";")->fetch_array(MYSQLI_NUM)[0] >= 1) {
            $error_msg = "You already annotated this sample!";
        } else {
            $validity = $conn_db->real_escape_string($_POST["validity"]);
            $novelty = $conn_db->real_escape_string($_POST["novelty"]);
            $generalFraming = $conn_db->real_escape_string($_POST["generalFraming"]);
            $specificFraming = $conn_db->real_escape_string($_POST["specificFraming"]);
            $time = time() - $_POST["timeStart"];
            $comments = $conn_db->real_escape_string($_POST["comments"]);
            $comments = (is_null($comments) or $comments == "") ? "no comment" : $comments;

            $query = "INSERT INTO CrowdSourceAnswer (annotator_ID, test_ID, timeInS, a_validity, a_novelty, a_issuespecificframe, a_genericmappedframe, a_comment) VALUES 
                                            (".$annotator_ID.", ".$sample_ID_commit.", ". $time .", ".$validity .", ".$novelty.", ".$specificFraming.", ".$generalFraming.", \"".$comments."\");";

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
        $general_frame = "nothing";
        $specific_frame = "nothing";
    } else {
        $conn_db = new mysqli("localhost", "philipp", $db_password, "philipp_umfrage");
        $stat = $conn_db->error . '<br>'. $conn_db->stat() .'<br>';

        $annotator_ID = intval($conn_db->real_escape_string($_GET["annotator_ID"]));
        
        $samples_done = $conn_db->query("SELECT COUNT(*) FROM CrowdSourceSamples WHERE EXISTS(SELECT * FROM CrowdSourceAnswer WHERE CrowdSourceAnswer.test_ID = CrowdSourceSamples.test_ID and annotator_ID = ". $annotator_ID .");")->fetch_array(MYSQLI_NUM);
        $samples_done = $samples_done === FALSE ? 0 : (is_null($samples_done) ? 1 : $samples_done[0]);
        $samples_total = $conn_db->query("SELECT COUNT(*) FROM CrowdSourceSamples")->fetch_array(MYSQLI_NUM);
        $samples_total =  ($samples_total === FALSE or is_null($samples_total)) ? 1 :  max(1, $samples_total[0]);

	    $result = $conn_db -> query("SELECT * FROM CrowdSourceSamples WHERE NOT EXISTS(SELECT * FROM CrowdSourceAnswer WHERE annotator_ID = ". $annotator_ID ." and CrowdSourceAnswer.test_ID = CrowdSourceSamples.test_ID) LIMIT 1;");
        
        if ($result === FALSE or $result->num_rows == 0) {
            $sample_ID = "-1";
            $topic = "WARNING";
            $premise = "SQL-Query failed";
            $conclusion1 = "Either there is a problem with the database";
            $conclusion2 = "... or you finished all the samples properly :)";
            $general_frame = "nothing";
            $specific_frame = "nothing";
        } else {
            $sample_ID = "-1";
            $topic = "...";
            $premise = "loading...";
            $conclusion1 = "loading...";
            $conclusion2 = ".loading...";
            $general_frame = "loading...";
            $specific_frame = "loading...";
            while ($row = $result->fetch_assoc()) {
                $sample_ID = (array_key_exists("test_ID", $row) or !is_null($row["test_ID"])) ? $row["test_ID"]: 0;
                $topic = $row["topic"];
                $premise = $row["premise"];
                $conclusion1 = $row["conclusion1"];
                $conclusion2 =  $row["conclusion2"];
                $general_frame =  is_null($row["genericmappedframe"]) ? "general" : $row["genericmappedframe"];
                $specific_frame =  (is_null($row["genericmappedframe"]) ? (is_null($row["genericinferredframe"]) ? "unspecific" : $row["genericinferredframe"])  : $row["genericmappedframe"]);
            }
        }
        
        $conn_db->close();
    }

    $topic = is_null($error_msg) ? $topic : $error_msg;
?>

<header>
    <title>You're in annotation mood (<?php echo $_GET["annotator_ID"] ?>)</title>
    <script>
            function Validity(l) {
                if (l == 'left') {
                    document.getElementById('left').style.backgroundColor = "#90EE90";
                    document.getElementById('right').style.backgroundColor = "yellow";
                } else if (l == 'right') {
                    document.getElementById('right').style.backgroundColor = "#90EE90";
                    document.getElementById('left').style.backgroundColor = "yellow";
                } else {
                    document.getElementById('left').style.backgroundColor = "yellow";
                    document.getElementById('right').style.backgroundColor = "yellow";
                }
            }

            function  Novelty(l) {
                if (l == 'left') {
                    document.getElementById('left').style.fontWeight = "bold";
                    document.getElementById('right').style.fontWeight = "normal";
                } else if(l == 'right') {
                    document.getElementById('right').style.fontWeight = "bold";
                    document.getElementById('left').style.fontWeight = "normal";
                } else {
                    document.getElementById('left').style.fontWeight = "normal";
                    document.getElementById('right').style.fontWeight = "normal";
                }
            }

            function  FrameSpec(l) {
                if (l == 'left') {
                    document.getElementById('left').style.borderWidth = "3px";
                    document.getElementById('right').style.borderWidth = "1px";
                } else if(l == 'right') {
                    document.getElementById('right').style.borderWidth = "3px";
                    document.getElementById('left').style.borderWidth = "1px";
                } else {
                    document.getElementById('left').style.borderWidth = "1px";
                    document.getElementById('right').style.borderWidth = "1px";
                }
            }

            function  FrameGen(l) {
                if (l == 'left') {
                    document.getElementById('left').style.borderStyle = "solid";
                    document.getElementById('right').style.borderStyle = "dashed";
                } else if(l == 'right') {
                    document.getElementById('right').style.borderStyle = "solid";
                    document.getElementById('left').style.borderStyle = "dashed";
                } else {
                    document.getElementById('left').style.borderStyle = "dashed";
                    document.getElementById('right').style.borderStyle = "dashed";
                }
            }
    </script>
</header>
<body style="text-align: center; max-width: 1000px; margin: auto;">
    <h1>Sample <?php echo $sample_ID; ?></h1>
    <progress id="progress_annotation" value="<?php echo $samples_done*1000/$samples_total ?>" max="1000"> <?php echo round($samples_done*100/$samples_total) ?>% </progress>
    <h2><?php echo $topic; ?></h2>
    <div style="width: 98%; background-color: lightgray; border: 1px solid black; border-radius: 20px; padding: 10px; margin-top: 12px; margin-bottom: 20px; display: inline-block;"><b>Premise:</b> <?php echo $premise; ?></div>
    <div style="display: flex; width: 100%;">
        <div id="left" style="width: 48%; background-color: yellow; border: 1px dashed black; border-radius: 20px; padding: 25px;"><b>Conclusion 1:</b> <?php echo $conclusion1; ?></div>
        <div id="right" style="width: 48%; background-color: yellow; border: 1px dashed black; border-radius: 20px; padding: 25px;"><b>Conclusion 2:</b> <?php echo $conclusion2; ?></div>
    </div>
    <h2>Let's rate ;)</h2>
    <form action="annotate.php?annotator_ID=<?php echo $_GET["annotator_ID"]; ?>" method="POST" autocomplete="off">
        <h3>Validity</h3>
        <ol>
            <li><input type="radio" value="-1" name="validity" required onclick="Validity('left');" title="Conclusion 1">Conclusion 1 fits better to the premise</li>
            <li><input type="radio" value="0" name="validity" required onclick="Validity('none');" title="NONE">Both are equally bad/ good (try to avoid this option - only in real ties)</li>
            <li><input type="radio" value="1" name="validity" required onclick="Validity('right');" title="Conclusion 2">Conclusion 2 fits better to the premise</li>
        </ol>
        <h3>Novelty</h3>
        <ol>
            <li><input type="radio" value="-1" name="novelty" required onclick="Novelty('left');" title="Conclusion 1">Conclusion 1 contains more novel (proper) information</li>
            <li><input type="radio" value="0" name="novelty" required onclick="Novelty('none');" title="NONE">Both are equally bad/ good (try to avoid this option - only in real ties)</li>
            <li><input type="radio" value="1" name="novelty" required onclick="Novelty('right');" title="Conclusion 2">Conclusion 2 contains more novel (proper) information</li>
        </ol>
        <h3> Generic perspective &raquo;<?php echo $general_frame; ?>&laquo;</h3>
        <ol>
            <li><input type="radio" value="-1" name="generalFraming" required onclick="FrameGen('left');" title="Conclusion 1">Conclusion 1 captures more of the generic perspective</li>
            <li><input type="radio" value="0" name="generalFraming" required onclick="FrameGen('none');" title="NONE">Both are equally bad/ good (try to avoid this option -only in real ties)</li>
            <li><input type="radio" value="1" name="generalFraming" required onclick="FrameGen('right');" title="Conclusion 2">Conclusion 2 captures more of the generic perspective</li>
        </ol>
        <h3>Specific perspective &raquo;<?php echo $specific_frame; ?>&laquo;</h3>
        <ol>
            <li><input type="radio" value="-1" name="specificFraming" required onclick="FrameSpec('left');" title="Conclusion 1">Conclusion 1 captures more of the specific perspective</li>
            <li><input type="radio" value="0" name="specificFraming" required onclick="FrameSpec('none');" title="NONE">Both are equally bad/ good (try to avoid this option - only in real ties)</li>
            <li><input type="radio" value="1" name="specificFraming" required onclick="FrameSpec('right');" title="Conclusion 2">Conclusion 2 captures more of the specific perspective</li>
        </ol>
        <textarea cols="80" rows="2" name="comments" maxlength="256" placeholder="Any questions/ comments to this sample? (optional)"></textarea>
        <br>
        <input type="hidden" value="<?php echo $sample_ID; ?>" name="sample_ID">
        <input type="hidden" value="<?php echo time(); ?>" name="timeStart">
        <br>
        <input type="submit" value=">>> Save & next >>>">
    </form>
    <hr>
    <div style="display: inline-block; width: 100%;">
        <h2>Instructions</h2>
        <p>We ask you to rate conclusions in a pair-wise manner in a discussion. To this end, you have for each argument the topic in the title, a premise (the explain- / give-reasons-part of an argument) and two conclusions (hopefully) matching the premise. But - which conclusion is more appropriate? Which conclusion fits the premise better?</p>

        <p>Premise --&gt; conclusion 1 ? OR Premise --&gt; conclusion 2?</p>

        <h4>To have differentiated voting, you have to decide on 3+1 aspects.</h4>

        <ol>
            <li><b>Validity:</b> Which conclusion is more reasonable / more appropriate? So, what is more likely? Conclusion 1 follows given the premise or conclusion 2?</li>
            <li><b>Novelty:</b> Which conclusion contains more novel information? Copying (or rephrasing) parts of the premise is too easy (should get no vote rather). Here we seek the conclusion with the most novel content, with the most conclusion-information-gain.</li>
            <li><b>Perspective &raquo;xxx&laquo;:</b> You can argue from different perspectives. For example, if you discuss raising taxes, you can emphasize the side of the tax-payers or the government's side or think about moral viewpoints. All these perspectives are called &raquo;Frames&laquo;. Here, you should rate how much the desired perspective occurs in the conclusions. A conclusion that fits more (obviously) into the desired perspective should gain the vote.</li>
        </ol>
        <h3>Positive and negative examples</h3>
        <h4>Positive (you should do it in such a way)</h4>
        <img src="../img/AZ/positive_example.jpg" width="50%">
        <ul>
            <li>differentiated voting: although conclusion 2 is the more trivial one (somewhat a rephrased interpretation of the premise), conclusion 1 is more novel and a real inference, hence it receives 2/4 votes</li>
            <li>you read each text carefully, thinking about it and hence, you noticed conclusion 2 is a little bit clearer about research and conclusion 1 captures more a big question of life</li>
        </ul>
        <h4>Negative (no, no, no...)</h4>
        <img src="../img/AZ/negative_example.jpg" width="50%">
        <ul>
            <li>lazy voting (here only for conclusion 2 or tie) - yes, sometimes one conclusion outperforms the other in <b>all</b> categories, but this is rare</li>
            <li>maybe you're thinking: "I'm a atheist, therefore nothing with God!" and therefore, you voted for conclusion 2 - however, we don't ask for your personal opinion about the topic, only stick to the premise and the conclusion - and the premise is a perfect base for conclusion 1.</li>
        </ul>
    </div>
</body>