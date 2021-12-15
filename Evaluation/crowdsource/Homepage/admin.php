<?php
$answer = "no request";

if($_POST) {
    $conn_db = new mysqli("localhost", "philipp", $_POST["pw"], "philipp_umfrage");

    $res = $conn_db->multi_query($_POST['query']);

    if($res === TRUE) {
        $answer = "<u>SQL command successfully!</u><br>";
        do {
            /* store the result set in PHP */
            if ($res_print = $conn_db->store_result()) {
                if($res_print === FALSE) {
                    $answer .= "executed...<br>";
                }
                while ($row = $res_print->fetch_row()) {
                    $answer .= join("|", $row) . "<br>";
                }
            }
            /* print divider */
            if ($conn_db->more_results()) {
                $answer .= "-----------------<br>";
            }
        } while ($conn_db->next_result());
    } else {
        $answer = $conn_db->error;
    }

    $conn_db->close();
}
?>

<header>
    <title>MySQL-Admin</title>
</header>
<body>
    <form action="admin.php" method="post" autocomplete="off">
        Password: <input type="password" name="pw" maxlength="50" required><br>
        <textarea name="query" wrap="hard" required rows="15" cols="80">SQL-Query</textarea><br>
        <button type="submit">Execute</button>
    </form>
    <hr>
    <?php echo $answer ?>
</body>