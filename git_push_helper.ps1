param(
    [string]$GitHubUser = "icezingza",
    [string]$RepoName = "namonexus-engine",
    [string]$CommitMsg = "Update: NamoNexus Engine v4.0.0"
)

# หยุดทำงานถ้าเจอ Error ร้ายแรง (ยกเว้น git commit กรณีไม่มีไฟล์แก้)
$ErrorActionPreference = "Continue"

$RepoUrl = "https://github.com/$GitHubUser/$RepoName.git"

Write-Host "🚀 NamoNexus Auto-Pusher" -ForegroundColor Cyan
Write-Host "--------------------------------"
Write-Host "Target: $RepoUrl"

# 1. ตรวจสอบและเริ่มระบบ Git (ถ้ายังไม่มี)
if (-not (Test-Path .git)) {
    Write-Host "[1/3] Initializing new repository..." -ForegroundColor Yellow
    git init
    git branch -M main
    git remote add origin $RepoUrl
}
else {
    Write-Host "[1/3] Repository found." -ForegroundColor Green
    # อัปเดต URL เผื่อมีการเปลี่ยนชื่อ Repo
    if ((git remote get-url origin) -ne $RepoUrl) {
        Write-Host "Updating remote URL..." -ForegroundColor Yellow
        git remote set-url origin $RepoUrl
    }
}

# 2. บันทึกไฟล์ (Stage & Commit)
Write-Host "[2/3] Staging and Committing files..." -ForegroundColor Yellow
git add .
git commit -m $CommitMsg

# 3. ส่งขึ้น GitHub (Push)
Write-Host "[3/3] Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ SUCCESS! View your code at: $RepoUrl" -ForegroundColor Green
}
else {
    Write-Host "`n❌ PUSH FAILED. Please check your internet or GitHub permissions." -ForegroundColor Red
}