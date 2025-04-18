<# 
  setup.ps1: Windows PowerShell用 環境構築スクリプト 
  実行前に「PowerShell を管理者権限で実行」するとスムーズ
#>

param(
  [string]$RepoURL = 'https://github.com/Az-san/EEG_prediction.git',
  [string]$RepoDir = 'EEG_prediction'
)

# 1. リポジトリ取得
if (-not (Test-Path $RepoDir)) {
    git clone $RepoURL
}
Set-Location $RepoDir

# 2. 仮想環境作成
python -m venv venv

# 3. 仮想環境有効化
#    実行ポリシーに引っかかる場合は：
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\venv\Scripts\Activate.ps1

# 4. pip 更新
pip install --upgrade pip

# 5. 本体用ライブラリ一気インストール
pip install `
  numpy pandas scipy matplotlib seaborn `
  tensorflow tensorflow-io `
  pyqtgraph PyQt5 pygame pyttsx3 websockets ipython

# 6. WebSocketビジュアライザ用ライブラリ
pip install -r show_websocket/source_controller/requirements.txt
pip install -r show_websocket/source_show/requirements.txt

