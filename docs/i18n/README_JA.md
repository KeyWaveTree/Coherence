# CudaBridge

**[English](README_EN.md)** | **日本語** | **[中文](README_ZH.md)** | **[한국어](../../README.md)**

Apple Silicon MacでThunderbolt/USB4経由の外部NVIDIA eGPUによるCUDA演算を可能にするオープンソースプロジェクト

## 概要

CudaBridgeは、Apple Silicon (M1/M2/M3/M4) ベースのMacで、USB4/Thunderboltを通じて接続された
外部NVIDIA GPUでCUDA演算を実行できるようにするソフトウェアスタックです。

## アーキテクチャ

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Applications (CUDA Code)                     │
├─────────────────────────────────────────────────────────────────────────┤
│   Python API (cudabridge.py)  │  CudaBridge Userspace Library            │
│   numpy互換GPU演算             │  (libcudabridge.dylib / API Layer)       │
├─────────────────────────────────────────────────────────────────────────┤
│   CLI Tool (cudabridge-cli)   │  CUDA Runtime Bridge                     │
│   GPUモニタリング/設定管理      │  (cudaMemcpy, cudaMalloc, kernel等)      │
├─────────────────────────────────────────────────────────────────────────┤
│   eGPU Safety Manager         │  GPU Driver Compatibility Layer           │
│   接続/切断/復旧管理           │  (NVIDIA GPU初期化、コマンドキュー、メモリ) │
├─────────────────────────────────────────────────────────────────────────┤
│   Logging System              │  PCIe Tunneling Over USB4/Thunderbolt     │
│   構造化ログ/ローテーション     │  (PCIeカプセル化、DMA処理、割り込み)       │
├─────────────────────────────────────────────────────────────────────────┤
│                       macOS DriverKit Extension                          │
│                    (USB4/Thunderboltハードウェアアクセス)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                         Hardware Layer                                   │
│            Apple Silicon ←→ USB4/TB4 ←→ eGPU Enclosure ←→ NVIDIA GPU    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 主要コンポーネント

### 1. GPU Driver CLI (`cudabridge-cli`)

nvidia-smiに類似したGPU制御・モニタリング用コマンドラインツールです。

**主要コマンド:**

| コマンド | 別名 | 説明 |
|---------|------|------|
| `info` | `i` | GPUデバイス詳細情報の表示 |
| `status` | `s` | eGPU接続状態とヘルスメトリクスの表示 |
| `connect` | `c` | eGPU接続（自動検出/互換性チェック） |
| `disconnect` | `dc` | eGPU安全切断（`--force`で強制切断も可） |
| `monitor` | `m` | リアルタイムGPUモニタリング（温度、消費電力、エラー率） |
| `config` | `cf` | GPU設定管理（クロック、電力、ファン、P-state） |
| `diag` | `d` | システム診断の実行 |
| `log` | `l` | ドライバログの表示 |
| `benchmark` | `b` | 転送・演算パフォーマンスベンチマーク |
| `reset` | `r` | GPUリセット（ソフト/ハード） |

**CLI使用例:**

```bash
# GPU情報の確認
cudabridge-cli info

# eGPU接続
cudabridge-cli connect

# リアルタイムモニタリング（500ms間隔）
cudabridge-cli monitor -i 500

# 設定の確認
cudabridge-cli config show

# クロック速度の変更
cudabridge-cli config clock 2100 1200

# JSON出力（スクリプト連携用）
cudabridge-cli info --json

# 安全切断
cudabridge-cli disconnect

# 強制切断（緊急時）
cudabridge-cli disconnect --force
```

### 2. eGPU接続安全管理

Thunderbolt/USB4を通じたeGPUの安全な接続、切断、エラー復旧を管理します。

**安全機能:**
- **互換性チェック**: NVIDIA GPU専用、最小帯域幅/電力の検証。非互換デバイス検出時にエラー通知後強制切断
- **自動復旧**: リンクエラー検出時、ソフトリセット → ハードリセット → 再接続の順で自動復旧（最大3回）
- **データ整合性**: CRC32ベースの転送データ検証
- **熱保護**: GPU温度が90°C（設定可能）を超えた場合の緊急停止
- **ハートビートモニタリング**: 定期的な接続状態確認、タイムアウト時の自動復旧
- **安全切断**: 実行中の操作完了待機 → リソース解放 → トンネル解除

### 3. ロギングシステム

構造化ロギングでGPUドライバのデバッグを支援します。

- 6段階ログレベル: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- カテゴリ別フィルタリング: GENERAL, DRIVER, EGPU, MEMORY, PCIE, USB4, CUDA, CLI, PYTHON
- ログファイル自動ローテーション（デフォルト10MB、5ファイル保持）
- スレッドセーフ（pthread mutex）

### 4. Python API (`cudabridge.py`)

PythonからCUDAライブラリなしで、numpyコードとほぼ同じ構文でeGPU CUDA演算を実行できます。

```python
import numpy as np
import cudabridge as cb

cb.init()

# データをGPUへ転送
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

gpu_a = cb.to_device(a)
gpu_b = cb.to_device(b)

# GPUで演算（Pythonの演算子がそのまま使えます！）
gpu_c = gpu_a + gpu_b          # 要素ごとの加算
gpu_d = gpu_a * gpu_b          # 要素ごとの乗算

# 結果をnumpyで取得
result = cb.from_device(gpu_c)
print(result)  # [6.0, 8.0, 10.0, 12.0]

# 行列積
mat_a = cb.to_device(np.random.rand(100, 200).astype(np.float32))
mat_b = cb.to_device(np.random.rand(200, 50).astype(np.float32))
mat_c = mat_a @ mat_b   # (100, 50) の結果

cb.shutdown()
```

### 5. USB4/Thunderbolt PCIeトンネリングドライバ
- Apple SiliconのUSB4コントローラとの直接通信
- PCIeトランザクションのUSB4プロトコルへのカプセル化
- DMA（ダイレクトメモリアクセス）処理
- ホットプラグ対応

### 6. NVIDIA GPU互換レイヤ
- GPU初期化とリセットシーケンス
- BAR（ベースアドレスレジスタ）メモリマッピング
- コマンド送信と完了処理
- 電力管理

### 7. CUDA Runtime Bridge
- CUDA Runtime API互換インターフェース
- カーネルのコンパイルと実行
- メモリ管理（ホスト ↔ デバイス）
- ストリームとイベントの同期

## システム要件

### ハードウェア
- Apple Silicon Mac (M1, M2, M3, M4 シリーズ)
- USB4またはThunderbolt 4ポート
- 互換性のあるeGPUエンクロージャ（Thunderbolt 3/4対応）
- NVIDIA GPU（Ampere、Ada Lovelace、または最新アーキテクチャ推奨）

### ソフトウェア
- macOS 13.0 (Ventura) 以降
- Xcode Command Line Tools
- CMake 3.20+
- Python 3.8+（Python API使用時）
- numpy（Python API使用時）

## ビルド方法

```bash
git clone https://github.com/KeyWaveTree/CudaBridge.git
cd CudaBridge
mkdir build && cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
sudo make install
```

## クイックスタート

```bash
cudabridge-cli connect        # eGPU接続
cudabridge-cli info            # GPU情報表示
cudabridge-cli status          # ステータス表示
cudabridge-cli diag            # 診断実行
cudabridge-cli disconnect      # 安全切断
```

## 学習ガイド

### ステップ1: CLIツールから始める
CLIツールはコーディングなしでGPUの状態確認や設定変更ができる最も簡単な出発点です。

### ステップ2: Python APIでGPU演算を体験
Python APIは既存のnumpyコードとほぼ同じ構文でGPU演算を実行します。

### ステップ3: C APIで低レベル制御
C APIを使えばメモリ管理、ストリーム、イベントなどを直接制御できます。

### ステップ4: eGPU安全管理の理解
eGPU接続の安全メカニズムを理解すれば、安定したアプリケーションを開発できます。

### ステップ5: ロギングシステムの活用
デバッグ時にロギングシステムを活用すれば、問題を素早く特定できます。

## 既知の制限事項

1. **macOS SIP**: System Integrity Protectionの無効化が必要な場合があります
2. **ドライバ署名**: 開発中は開発者署名が必要です
3. **帯域幅**: USB4の理論的最大帯域幅は40Gbps（PCIe 3.0 x4レベル）
4. **互換性**: すべてのCUDA機能がサポートされるとは限りません
5. **シミュレーションモード**: 実際のeGPUなしでも開発/テストが可能です

## コントリビューション

コントリビューションを歓迎します！Pull Requestを送る前にIssueを作成してください。

## 謝辞と出典表記の推奨 (Attribution Recommendation)

このプロジェクトはMITライセンスの下で自由に利用・貢献できるよう公開されています。
法的な義務ではありませんが、オープンソースエコシステムの好循環と制作者のモチベーション向上のため、
以下のガイドラインを守っていただくことを推奨します。

> 「このプロジェクトを活用して素晴らしい成果物を作られたなら、出典を残してください。大きなやりがいとモチベーションになります！」

- **出典表記と謝辞**: このコードを活用して製品やサービスを作られた場合、製品紹介ページ、ブログ、
  発表資料、またはソースコードのコメントに原作者（CudaBridge Contributors）への謝辞を
  記載していただけると幸いです。

- **プロジェクトのフォーク時**: このプロジェクトをベースにフォークまたは発展させて新しいプロジェクトを
  配布される場合、このプロジェクトが基盤となったことを明示的に記載してください。

皆さまの温かい一言が、開発者がプロジェクトを発展させ続ける最大の原動力となります。ありがとうございます！

## ライセンス

MIT License - 詳細は [LICENSE](../../LICENSE) を参照

## 免責事項

このプロジェクトは実験的であり、教育目的で提供されています。本番環境での使用は推奨しません。
Apple、NVIDIA、CUDAは各社の商標です。
