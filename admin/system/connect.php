<?php 

try {
    $db=new PDO("mysql:host=localhost;dbname=ecommerce;charset=utf8", 'root','');
    //echo "Connection Successful";

} catch (PDOException $e) {
    echo $e -> getMessage();
}
?>