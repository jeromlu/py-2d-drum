[CmdletBinding(PositionalBinding=$false)]

Param(
    # If specified, the script only removes the build outputs and exits.
    [Switch]$Clean,
    # If specified, the produced pdf file is opened in Sumatra pdf.
    [Switch]$ShowPdf
    #[Switch]$CreateHtml
)

# Quit on cmdlet errors.
$ErrorActionPreference = 'Stop'

$buildDirName = "build"
# Name of the file that will be compiled. Output files will also be named with the same name.
#$fileName = "brachytherapyNotes"
$fileName = "brachy_notes_main"
$BUILD_DIR = "$PSScriptRoot\$buildDirName"

function Remove-ByForce {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue @Args
}

if ($Clean) {
    Remove-ByForce $BUILD_DIR
    exit
}

# Build.
New-Item -Force -ItemType directory -Path $BUILD_DIR
$sourceFile = "$PSScriptRoot/src/$fileName.tex"
try {
    $sourceFile
    $buildDirName
    latexmk -pdf -lualatex -jobname="./$buildDirName/%A" $sourceFile
    # Needed only for sumatraPDF.
    Push-Location -Path $BUILD_DIR
    if ($CreateHtml) {
        "Create html. Not yet implemented."
        #mk4ht htlatex $sourceFile 'xhtml,charset=utf-8,pmathml' ' -cunihtf -utf8 -cvalidate'
    }
    if ($ShowPdf) {
        SumatraPDF.exe "$fileName.pdf"
    }
}
finally {
    Pop-Location
}
