# Renders the Kor Studio logo (same geometry as assets/logo.svg) to
# assets/logo.png and assets/kor.ico using GDI+. Windows only.
#   powershell -ExecutionPolicy Bypass -File scripts/make_icon.ps1
Add-Type -AssemblyName System.Drawing
$assets = Join-Path (Split-Path $PSScriptRoot -Parent) 'assets'

function New-LogoBitmap([int]$size) {
    $bmp = New-Object System.Drawing.Bitmap $size, $size, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'
    $g.PixelOffsetMode = 'HighQuality'
    $g.Clear([System.Drawing.Color]::Transparent)
    $k = $size / 256.0
    $g.ScaleTransform($k, $k)
    $dark = [System.Drawing.Color]::FromArgb(255, 17, 13, 28)

    # rounded background tile
    $r = 56; $d = 2 * $r
    $tile = New-Object System.Drawing.Drawing2D.GraphicsPath
    $tile.AddArc(0, 0, $d, $d, 180, 90)
    $tile.AddArc(256 - $d, 0, $d, $d, 270, 90)
    $tile.AddArc(256 - $d, 256 - $d, $d, $d, 0, 90)
    $tile.AddArc(0, 256 - $d, $d, $d, 90, 90)
    $tile.CloseFigure()
    $bg = New-Object System.Drawing.SolidBrush $dark
    $g.FillPath($bg, $tile)

    # ember
    $ember = New-Object System.Drawing.Drawing2D.GraphicsPath
    $ember.AddBezier(128, 26, 150, 70, 202, 100, 202, 154)
    $ember.AddArc(54, 80, 148, 148, 0, 180)
    $ember.AddBezier(54, 154, 54, 100, 106, 70, 128, 26)
    $ember.CloseFigure()
    $rect = New-Object System.Drawing.RectangleF 0, 24, 256, 208
    $grad = New-Object System.Drawing.Drawing2D.LinearGradientBrush $rect, ([System.Drawing.Color]::White), ([System.Drawing.Color]::Black), 90.0
    $blend = New-Object System.Drawing.Drawing2D.ColorBlend 3
    $blend.Colors = @([System.Drawing.Color]::FromArgb(255, 196, 168, 255), [System.Drawing.Color]::FromArgb(255, 139, 92, 246), [System.Drawing.Color]::FromArgb(255, 109, 63, 217))
    $blend.Positions = @(0.0, 0.55, 1.0)
    $grad.InterpolationColors = $blend
    $g.FillPath($grad, $ember)

    # network
    $pen = New-Object System.Drawing.Pen $dark, 9
    $pen.StartCap = 'Round'; $pen.EndCap = 'Round'
    $nodes = @(@(128, 112), @(94, 172), @(162, 172))
    $g.DrawLine($pen, 128, 112, 94, 172)
    $g.DrawLine($pen, 128, 112, 162, 172)
    $g.DrawLine($pen, 94, 172, 162, 172)
    $hi = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(255, 219, 204, 255))
    foreach ($n in $nodes) {
        $g.FillEllipse($bg, $n[0] - 16, $n[1] - 16, 32, 32)
        $g.FillEllipse($hi, $n[0] - 6.5, $n[1] - 6.5, 13, 13)
    }
    $g.Dispose()
    return $bmp
}

function Get-PngBytes($bmp) {
    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    return ,$ms.ToArray()
}

$big = New-LogoBitmap 512
$big.Save((Join-Path $assets 'logo.png'), [System.Drawing.Imaging.ImageFormat]::Png)

# ICO container holding PNG-compressed images.
$sizes = 16, 32, 48, 64, 128, 256
$images = foreach ($s in $sizes) { ,(Get-PngBytes (New-LogoBitmap $s)) }
$out = New-Object System.IO.MemoryStream
$w = New-Object System.IO.BinaryWriter $out
$w.Write([uint16]0); $w.Write([uint16]1); $w.Write([uint16]$sizes.Count)
$offset = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $dim = if ($sizes[$i] -ge 256) { 0 } else { $sizes[$i] }
    $w.Write([byte]$dim); $w.Write([byte]$dim); $w.Write([byte]0); $w.Write([byte]0)
    $w.Write([uint16]1); $w.Write([uint16]32)
    $w.Write([uint32]$images[$i].Length); $w.Write([uint32]$offset)
    $offset += $images[$i].Length
}
foreach ($img in $images) { $w.Write($img) }
[System.IO.File]::WriteAllBytes((Join-Path $assets 'kor.ico'), $out.ToArray())
Write-Host "wrote logo.png and kor.ico to $assets"
