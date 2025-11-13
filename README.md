# NovelAI API for ComfyUI

NovelAI APIを使用してComfyUIで画像生成を行うためのカスタムノードです。

## 機能

- **NovelAI Image Generator**: テキストプロンプトから画像を生成
- **NovelAI Image to Image**: 既存の画像をベースに新しい画像を生成
- **NovelAI Character Prompt**: V4/V4.5モデル用の個別キャラクター定義
- **NovelAI Character Prompt Combine**: 複数のキャラクタープロンプトを結合

## インストール

1. このリポジトリをComfyUIのcustom_nodesディレクトリにクローンまたはダウンロード:
```bash
cd ComfyUI/custom_nodes
git clone <repository-url> NovelAI_API
```

2. 依存パッケージをインストール:
```bash
cd NovelAI_API
pip install -r requirements.txt
```

3. ComfyUIを再起動

## 使い方

### NovelAI Image Generator

テキストプロンプトから画像を生成するノードです。

**入力パラメータ:**
- `api_key`: NovelAI APIキー（必須）
- `prompt`: ポジティブプロンプト
- `negative_prompt`: ネガティブプロンプト
- `model`: 使用するモデル（V4.5 Curated/Full、V4 Full、V3、V2、V1）
- `width`: 画像の幅（64-2048、64刻み）
- `height`: 画像の高さ（64-2048、64刻み）
- `steps`: サンプリングステップ数（1-50）
- `scale`: CFGスケール（0.0-10.0）
- `sampler`: サンプラー選択
- `seed`: ランダムシード値
- `n_samples`: 生成枚数（1-4）
- `noise_schedule`: ノイズスケジュール（オプション）
- `character_prompts`: キャラクタープロンプトリスト（オプション、V4/V4.5のみ）

**出力:**
- `IMAGE`: 生成された画像（ComfyUI IMAGE形式）

### NovelAI Image to Image

既存の画像をベースに新しい画像を生成するノードです。

**入力パラメータ:**
- `api_key`: NovelAI APIキー（必須）
- `image`: 入力画像（ComfyUI IMAGE形式）
- `prompt`: ポジティブプロンプト
- `negative_prompt`: ネガティブプロンプト
- `model`: 使用するモデル
- `strength`: 変換強度（0.0-1.0）
- `noise`: ノイズ量（0.0-1.0）
- `steps`: サンプリングステップ数
- `scale`: CFGスケール
- `sampler`: サンプラー選択
- `seed`: ランダムシード値
- `character_prompts`: キャラクタープロンプトリスト（オプション、V4/V4.5のみ）

**出力:**
- `IMAGE`: 生成された画像（ComfyUI IMAGE形式）

### NovelAI Character Prompt（V4/V4.5専用）

V4/V4.5モデルで複数のキャラクターを個別に配置するためのノードです。

**入力パラメータ:**
- `prompt`: キャラクターのプロンプト（必須）
- `position`: キャラクターの位置（必須）
  - `AUTO`: 自動配置（結合時に自動的に位置が割り当てられます）
    - 割り当て順: A1 → E5 → C3 → A5 → E1
    - 既に使用されている位置は自動的にスキップされます
  - `A1`～`E5`: 5x5グリッド上の位置を明示的に指定
    - A行～E行（上から下）
    - 1列～5列（左から右）
- `character_negative_prompt`: キャラクター固有のネガティブプロンプト（オプション）
  - このキャラクターに適用される除外要素を指定
  - 例: "bad hands, extra fingers"
- `enabled`: このキャラクターを有効化（オプション、デフォルト: True）

**出力:**
- `NOVEL_AI_CHARACTER`: キャラクター定義データ

**使用例:**
```
Character Prompt 1
  - prompt: "1girl, blonde hair, blue eyes"
  - position: A1
  - character_negative_prompt: "bad anatomy"
    ↓
Character Prompt 2
  - prompt: "1boy, black hair, green eyes"
  - position: E5
  - character_negative_prompt: "extra limbs"
    ↓
Character Prompt Combine → Generator ノード
```

各キャラクターに個別のネガティブプロンプトを設定することで、キャラクターごとに異なる除外要素を指定できます。

### NovelAI Character Prompt Combine

複数のキャラクタープロンプトを結合して、メインの生成ノードに渡すためのノードです。

**入力パラメータ:**
- `character_1`～`character_5`: キャラクタープロンプト（オプション、最大5個）

**出力:**
- `NOVEL_AI_CHARACTER_LIST`: キャラクターリスト

**AUTO位置の自動割り当て:**
このノードは、`position: AUTO`で設定されたキャラクターに対して、自動的に位置を割り当てます：
- 割り当て順序: A1（左上）→ E5（右下）→ C3（中央）→ A5（右上）→ E1（左下）
- 明示的に位置が指定されているキャラクターは優先され、AUTO位置はそれらを避けて配置されます
- 例: キャラ1（A1指定）、キャラ2（AUTO）、キャラ3（AUTO）の場合
  - キャラ1: A1
  - キャラ2: E5（A1はスキップ）
  - キャラ3: C3

**使用方法:**
1. `NovelAI Character Prompt`ノードで個別のキャラクターを定義
2. `NovelAI Character Prompt Combine`ノードで結合
3. メインの`NovelAI Image Generator`または`NovelAI Image to Image`ノードの`character_prompts`入力に接続

**注意:**
- この機能はV4/V4.5モデルでのみ使用できます
- 位置を指定する場合、`use_coords`が自動的に有効になります
- base_captionと組み合わせて、複雑なマルチキャラクター画像を生成できます
- `character_negative_prompt`は各キャラクター固有のネガティブプロンプトです。メインの`negative_prompt`とは別に処理されます

## APIキーの取得

NovelAI APIを使用するには、有効なNovelAIアカウントとAPIキーが必要です。

1. [NovelAI](https://novelai.net/)にログイン
2. アカウント設定からAPIキーを取得
3. ノードの`api_key`パラメータに設定

## 対応サンプラー

### 標準サンプラー
- k_euler
- k_euler_ancestral
- k_dpm_2
- k_dpm_2_ancestral
- k_lms
- k_dpmpp_2s_ancestral
- k_dpmpp_sde
- k_dpmpp_2m
- k_dpmpp_2m_sde
- k_dpmpp_3m_sde
- k_dpm_adaptive
- k_dpm_fast
- plms
- ddim
- ddim_v3

### V4/V4.5専用サンプラー
- **nai_smea**: SMEA（Sample Masking and Entropy Adaptive）- V4/V4.5モデルに最適化
- **nai_smea_dyn**: SMEA Dynamic - より動的なサンプリング

**注意:** V4/V4.5モデルでは、これらのサンプラーが推奨されます。内部的にk_euler_ancestralサンプラーとSMEAフラグを使用します。

## 対応モデル

### V4.5シリーズ（最新）
- **nai-diffusion-4-5-curated**: V4.5 Curated（推奨・高品質データセット）
- **nai-diffusion-4-5-full**: V4.5 Full（完全データセット）

### V4シリーズ
- **nai-diffusion-4-full**: V4 Full

### V3以前
- **nai-diffusion-3**: V3（安定版）
- **nai-diffusion-2**: V2
- **nai-diffusion**: V1

## 注意事項

- APIキーは安全に管理してください
- API使用には NovelAI サブスクリプションが必要です
- 生成枚数やパラメータによってはAPIクレジットを多く消費します
- ネットワーク接続が必要です

### V4/V4.5モデルを使用する場合

- V4/V4.5モデルでは、内部的に構造化されたプロンプト形式（v4_prompt）を使用します
- V3以前のモデルとは異なるパラメータ構造が自動的に適用されます
- nai_smeaまたはnai_smea_dynサンプラーの使用を推奨します

## トラブルシューティング

### エラー: "APIキーが設定されていません"
- `api_key`パラメータに有効なNovelAI APIキーを入力してください

### エラー: "NovelAI API リクエストエラー"
- ネットワーク接続を確認してください
- APIキーが有効か確認してください
- NovelAIのサーバーステータスを確認してください

### エラー: "生成された画像が見つかりませんでした"
- リクエストパラメータが正しいか確認してください
- モデル名が正しいか確認してください

## ライセンス

MIT License

## 貢献

バグ報告や機能リクエストは、GitHubのIssuesでお願いします。

## 参考リンク

- [NovelAI公式サイト](https://novelai.net/)
- [NovelAI API ドキュメント](https://image.novelai.net/docs/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
