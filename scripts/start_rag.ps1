# 学习导读｜Windows 本地演示启动器：检查依赖、设置演示配置、启动后端/前端并等待健康检查。
# 演示模式使用 SQLite 和 deterministic Embedding，并关闭 Milvus；不能用它证明真实向量检索已部署成功。
# UseConfiguredLLM 仅控制是否启用已配置生成模型，其他演示配置仍保持此脚本设置。
# 脚本参数：可选启用生成模型，以及指定两个本机端口；此处只定义参数，不启动服务。
param(
    [switch]$UseConfiguredLLM,
    [switch]$ResetDemoData,
    [ValidateRange(1, 65535)]
    [int]$BackendPort = 8000,
    [ValidateRange(1, 65535)]
    [int]$FrontendPort = 3000
)

$ErrorActionPreference = "Stop"

# 以脚本所在目录定位项目，避免从其他目录启动时路径解析错误。
$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDirectory = Join-Path $projectRoot "backend"
$frontendDirectory = Join-Path $projectRoot "frontend"
$pythonPath = Join-Path $backendDirectory ".venv\Scripts\python.exe"
$dataDirectory = Join-Path $projectRoot ".data"
$databasePath = Join-Path $dataDirectory "devflow-local-demo.db"
$backendOutputLog = Join-Path $dataDirectory "devflow-backend.out.log"
$backendErrorLog = Join-Path $dataDirectory "devflow-backend.err.log"
$frontendOutputLog = Join-Path $dataDirectory "devflow-frontend.out.log"
$frontendErrorLog = Join-Path $dataDirectory "devflow-frontend.err.log"

# 只检查已安装依赖；本脚本不会自动创建虚拟环境或执行 npm install。
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "未找到 backend/.venv。请先创建虚拟环境并安装 backend/requirements.txt。"
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendDirectory "node_modules") -PathType Container)) {
    throw "未找到 frontend/node_modules。请先在 frontend 目录执行 npm install。"
}

New-Item -ItemType Directory -Path $dataDirectory -Force | Out-Null

# 演示仓库是脚本生成的合成数据，不是远程 GitHub checkout。
# 需要排除上一次运行残留的数据时，在启动前显式删除这份隔离 SQLite；不会触碰项目的 PostgreSQL。
if ($ResetDemoData -and (Test-Path -LiteralPath $databasePath -PathType Leaf)) {
    try {
        Remove-Item -LiteralPath $databasePath -Force
    }
    catch {
        throw "无法重置演示 SQLite：$databasePath。请先停止仍在使用 8100/8000 后端的进程，再重试。"
    }
}

$databaseUrlPath = $databasePath.Replace("\", "/")
# 这些环境变量传给后续子进程，覆盖 .env；不会把 .env 文件本身改成演示配置。
$env:DATABASE_URL = "sqlite:///$databaseUrlPath"
$env:MILVUS_ENABLED = "false"
$env:EMBEDDING_PROVIDER = "deterministic"
$env:EMBEDDING_MODEL = "deterministic-demo"
$env:EMBEDDING_DIMENSIONS = "128"
$env:AUTO_SYNC_ENABLED = "false"
$env:DEMO_MODE = "true"
$env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:$BackendPort"
$env:PYTHONPATH = $backendDirectory

# 默认关闭模型生成与 Judge，避免把离线演示误认为在线模型回答。
if (-not $UseConfiguredLLM) {
    $env:LLM_API_KEY = ""
    $env:RAG_LLM_ENABLED = "false"
    $env:RAGAS_ENABLED = "false"
}

$backendProcess = $null
$frontendProcess = $null

try {
    # 后台隐藏窗口启动 Uvicorn，保存进程对象和独立日志，以便检查失败及清理。
    $backendProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort") `
        -WorkingDirectory $backendDirectory `
        -RedirectStandardOutput $backendOutputLog `
        -RedirectStandardError $backendErrorLog `
        -WindowStyle Hidden `
        -PassThru

    # 通过 /health 探测就绪，而非启动进程后立即假设 API 已可用。
    $backendReady = $false
    for ($attempt = 0; $attempt -lt 60; $attempt += 1) {
        if ($backendProcess.HasExited) {
            throw "后端进程提前退出，请检查 $backendErrorLog。"
        }
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 2 | Out-Null
            $backendReady = $true
            break
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $backendReady) {
        throw "后端在 30 秒内没有就绪，请检查 $backendErrorLog。"
    }

    # 演示数据脚本复用后端导入路径；finally 恢复调用者原来的工作目录。
    Push-Location $backendDirectory
    try {
        & $pythonPath (Join-Path $projectRoot "scripts\seed_demo_data.py")
        if ($LASTEXITCODE -ne 0) {
            throw "演示数据初始化失败。"
        }
    }
    finally {
        Pop-Location
    }

    $demoRepositories = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/api/repos" -TimeoutSec 5
    $demoRepository = $demoRepositories | Where-Object { $_.full_name -eq "course-demo/devflow-sample-api" } | Select-Object -First 1
    if (-not $demoRepository) {
        throw "没有找到初始化后的演示仓库。"
    }
    # 向量关闭时停用演示库的分数阈值，便于展示关键词结果；这是降级演示，不是完整混合召回。
    Invoke-RestMethod `
        -Method Patch `
        -Uri "http://127.0.0.1:$BackendPort/api/rag/repositories/$($demoRepository.id)/config" `
        -ContentType "application/json" `
        -Body '{"retrieval_method":"hybrid","score_threshold_enabled":false}' `
        -TimeoutSec 10 | Out-Null

    # 前端以指定端口运行 Next.js 开发服务器，API 地址已由上面环境变量设置。
    $frontendProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "-p", "$FrontendPort") `
        -WorkingDirectory $frontendDirectory `
        -RedirectStandardOutput $frontendOutputLog `
        -RedirectStandardError $frontendErrorLog `
        -WindowStyle Hidden `
        -PassThru

    # 页面首次编译可能需要时间，继续轮询 HTTP 状态直到就绪或超时。
    $frontendReady = $false
    for ($attempt = 0; $attempt -lt 120; $attempt += 1) {
        if ($frontendProcess.HasExited) {
            throw "前端进程提前退出，请检查 $frontendErrorLog。"
        }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$FrontendPort" -TimeoutSec 2 | Out-Null
            $frontendReady = $true
            break
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $frontendReady) {
        throw "前端在 60 秒内没有就绪，请检查 $frontendErrorLog。"
    }

    Write-Host "DevFlow AI 本地演示已启动："
    Write-Host "  前端：http://127.0.0.1:$FrontendPort"
    Write-Host "  后端：http://127.0.0.1:$BackendPort"
    Write-Host "  后端 PID：$($backendProcess.Id)"
    Write-Host "  前端 PID：$($frontendProcess.Id)"
    Write-Host "  停止命令：Stop-Process -Id $($backendProcess.Id),$($frontendProcess.Id)"
    Write-Host "说明：该脚本关闭 Milvus并停用演示库的分数阈值，只用于页面、结构化数据和关键词检索演示；向量检索状态会显示 degraded。"
}
catch {
    # 启动链路失败时只尝试清理本脚本已创建且仍存活的进程。
    if ($frontendProcess -and -not $frontendProcess.HasExited) {
        Stop-Process -Id $frontendProcess.Id
    }
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id
    }
    throw
}
