# ============================================================================
# TimeGAN GPU - Complete Setup (Run in NEW PowerShell after restart)
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TimeGAN GPU Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$ENV_NAME = "timegan-gpu"
$CONDA_PATH = "D:\Apps\Anaconda\Scripts\conda.exe"

# Check if environment already exists
$envExists = & $CONDA_PATH env list | Select-String $ENV_NAME
if ($envExists) {
    Write-Host "`nEnvironment '$ENV_NAME' already exists!" -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove and recreate it? (y/n)"
    if ($response -eq 'y') {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        & $CONDA_PATH env remove -n $ENV_NAME -y
    } else {
        Write-Host "Keeping existing environment. Exiting..." -ForegroundColor Green
        exit 0
    }
}

Write-Host "`n[1/6] Creating conda environment with Python 3.9..." -ForegroundColor Yellow
& $CONDA_PATH create -n $ENV_NAME python=3.9 -y
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR creating environment" -ForegroundColor Red; exit 1 }

Write-Host "`n[2/6] Installing CUDA libraries..." -ForegroundColor Yellow
& $CONDA_PATH install -n $ENV_NAME -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR installing CUDA" -ForegroundColor Red; exit 1 }

Write-Host "`n[3/6] Installing TensorFlow via pip..." -ForegroundColor Yellow
& $CONDA_PATH run -n $ENV_NAME pip install "tensorflow==2.10.0"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR installing TensorFlow" -ForegroundColor Red; exit 1 }

Write-Host "`n[4/6] Installing data science packages..." -ForegroundColor Yellow
& $CONDA_PATH install -n $ENV_NAME -c conda-forge pandas numpy matplotlib seaborn scikit-learn scipy tqdm -y
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR installing packages" -ForegroundColor Red; exit 1 }

Write-Host "`n[5/6] Installing Jupyter..." -ForegroundColor Yellow
& $CONDA_PATH install -n $ENV_NAME -c conda-forge jupyter ipykernel -y
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR installing Jupyter" -ForegroundColor Red; exit 1 }

Write-Host "`n[6/6] Registering Jupyter kernel..." -ForegroundColor Yellow
& $CONDA_PATH run -n $ENV_NAME python -m ipykernel install --user --name $ENV_NAME --display-name "Python (TimeGAN-GPU)"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR registering kernel" -ForegroundColor Red; exit 1 }

Write-Host "`n[7/7] Verifying GPU detection..." -ForegroundColor Yellow
& $CONDA_PATH run -n $ENV_NAME python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'\nGPU Available: {len(gpus) > 0}'); print(f'Number of GPUs: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus] if gpus else print('  No GPU detected')"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  ✅ Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nTo use GPU in VS Code:" -ForegroundColor Cyan
Write-Host "1. Open: 02_timegan_copy_GPU.ipynb" -ForegroundColor White
Write-Host "2. Click 'Select Kernel' (top-right)" -ForegroundColor White
Write-Host "3. Choose: Python (TimeGAN-GPU)" -ForegroundColor White
Write-Host "4. Run Cell 2 to verify GPU is detected" -ForegroundColor White
Write-Host "5. Train with GPU acceleration! 🚀" -ForegroundColor White

Write-Host "`nTo activate environment manually:" -ForegroundColor Cyan
Write-Host "  conda activate $ENV_NAME" -ForegroundColor White

Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
