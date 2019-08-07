<?php


//Ejecutar script python y guardar print en $output
/*
ob_start();

passthru('C:\Users\Administrador\AppData\Local\Programs\Python\Python36\python.exe C:\AppServ\www\CNNWEB\scriptpython.py');

$output = ob_get_clean(); 
echo $output;
*/

//$orden = system( "C:\Users\Administrador\AppData\Local\Programs\Python\Python36\python.exe C:\AppServ\www\CNNWEB\scriptpython.py >>out.txt 2>>error.txt &" );
$cmd="C:\Users\Administrador\AppData\Local\Programs\Python\Python36\python.exe C:\AppServ\www\CNNWEB\scriptpython.py >>out.txt 2>>error.txt";
pclose(popen("start /B ". $cmd, "r"));
echo "jlhlj";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////77

//Insertar $valor en base de datos
/*include("conexionBD.php");

$valor='00001110';
    
$con=mysql_connect($host,$user,$pw)or die("problemas al conectar con BD");
mysql_select_db($db,$con)or die("problemas al conectar con bd");
mysql_query("INSERT INTO estado (estadoBancas) VALUES ('$valor')",$con);*/
////////////////////////////////////////////////////////////////////////


?>