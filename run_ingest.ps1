# run_ingest.ps1
# Sets up HADOOP_HOME and PATH for this session, shows diagnostics, then runs the ingestion script.
param(
    [string]$JavaHome = "",
    [switch]$Persist
)

# Ensure script runs from repo root
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# Set HADOOP_HOME to repo hadoop_home
$hadoopHome = Join-Path $repoRoot "hadoop_home"
if (-Not (Test-Path $hadoopHome)) {
    Write-Error "hadoop_home not found at $hadoopHome"
    exit 1
}
$env:HADOOP_HOME = $hadoopHome
$env:PATH = "$env:HADOOP_HOME\bin;$env:PATH"

# Optionally set JAVA_HOME
if ($JavaHome -ne "") {
    if (-Not (Test-Path $JavaHome)) {
        Write-Warning "Provided JAVA_HOME path does not exist: $JavaHome"
    } else {
        $env:JAVA_HOME = $JavaHome
        $env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
    }
}

Write-Host "== Environment diagnostics =="
Write-Host "HADOOP_HOME: $env:HADOOP_HOME"
Write-Host "winutils present:" (Test-Path (Join-Path $env:HADOOP_HOME "bin\winutils.exe"))

Write-Host "\njava -version output:"
try { java -version 2>&1 | ForEach-Object { Write-Host $_ } } catch { Write-Warning "java not found on PATH" }

Write-Host "\nwhere.exe winutils.exe:"
try { where.exe winutils.exe 2>&1 | ForEach-Object { Write-Host $_ } } catch { Write-Warning "winutils not found by where.exe" }

# Create a small temporary Python file to check Spark/Hadoop versions (avoid PowerShell heredoc issues)
$pyFile = Join-Path $env:TEMP "check_spark_hadoop_temp.py"
@'
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
print("Spark version:", spark.version)
try:
    print("Hadoop version:", spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion())
except Exception as e:
    print("Hadoop version: error", e)
spark.stop()
'@ | Out-File -FilePath $pyFile -Encoding utf8

Write-Host "\n== Spark & Hadoop versions (Python check) =="
python $pyFile
Remove-Item $pyFile -ErrorAction SilentlyContinue

Write-Host "\n== Running ingestion script =="
python .\src\M1_Ingestion\ingest_data.py

if ($Persist) {
    Write-Host "\nPersisting HADOOP_HOME and appending hadoop_home\bin to user PATH (setx)"
    setx HADOOP_HOME $env:HADOOP_HOME | Out-Null
    $oldPath = [Environment]::GetEnvironmentVariable("Path","User")
    if ($oldPath -notlike "*$($env:HADOOP_HOME)\bin*") {
        $newPath = "$oldPath;$env:HADOOP_HOME\bin"
        setx Path $newPath | Out-Null
    }
    if ($env:JAVA_HOME) { setx JAVA_HOME $env:JAVA_HOME | Out-Null }
    Write-Host "Persisted. Open a new shell to pick up changes."
}
